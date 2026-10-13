#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <random>
#include <ctype.h>
#include <chrono>

using namespace std;

const int D = 128;

struct WordInfo {
    int id;
    int doc_freq;
    float idf;
    vector<float> context_vec;
    vector<int> pos_idx;
    vector<int> neg_idx;
    string word;
};

struct Package {
    string name;
    string desc;
    vector<int> tokens;
    vector<float> embedding;
    float norm;
};

vector<Package> packages;
unordered_map<string, WordInfo> vocab_map;
vector<WordInfo*> vocab;

void tokenize(const string& text, vector<int>& tokens, unordered_set<int>& unique_tokens, bool add_to_vocab) {
    string w = "";
    auto process_word = [&](const string& word) {
        if (word.empty()) return;
        if (add_to_vocab) {
            if (vocab_map.find(word) == vocab_map.end()) {
                WordInfo wi;
                wi.id = vocab_map.size();
                wi.doc_freq = 0;
                wi.idf = 0.0f;
                wi.context_vec.resize(D, 0.0f);
                vocab_map[word] = wi;
            }
            int id = vocab_map[word].id;
            tokens.push_back(id);
            unique_tokens.insert(id);
        } else {
            if (vocab_map.find(word) != vocab_map.end()) {
                tokens.push_back(vocab_map[word].id);
                unique_tokens.insert(vocab_map[word].id);
            }
        }
    };

    for (char c : text) {
        if (isalnum(c)) {
            w += tolower(c);
        } else {
            process_word(w);
            w = "";
        }
    }
    process_word(w);
}

void load_packages() {
    auto read_from = [&](const string& cmd) {
        FILE* f = popen(cmd.c_str(), "r");
        if (!f) return;
        char buffer[4096];
        string current_pkg = "";
        string current_desc = "";
        bool in_desc = false;

        while (fgets(buffer, sizeof(buffer), f)) {
            string line = buffer;
            if (line.empty()) continue;
            if (line.back() == '\n') line.pop_back();
            if (line.back() == '\r') line.pop_back();

            if (line.empty()) {
                if (!current_pkg.empty()) {
                    Package p;
                    p.name = current_pkg;
                    p.desc = current_desc;
                    packages.push_back(p);
                    current_pkg = "";
                    current_desc = "";
                }
                in_desc = false;
                continue;
            }

            if (line.rfind("Package: ", 0) == 0) {
                current_pkg = line.substr(9);
                in_desc = false;
            } else if (line.rfind("Description: ", 0) == 0 || line.rfind("Description-en: ", 0) == 0) {
                size_t colon = line.find(':');
                if (colon != string::npos && colon + 2 < line.size()) {
                    current_desc = line.substr(colon + 2);
                }
                in_desc = true;
            } else if (in_desc && line.size() > 0 && line[0] == ' ') {
                current_desc += " " + line.substr(1);
            } else {
                in_desc = false;
            }
        }
        if (!current_pkg.empty()) {
            Package p;
            p.name = current_pkg;
            p.desc = current_desc;
            packages.push_back(p);
        }
        pclose(f);
    };

    read_from("apt-cache dumpavail 2>/dev/null");
    if (packages.empty()) {
        read_from("cat /var/lib/apt/lists/*_Packages 2>/dev/null");
    }
}

void build_embeddings() {
    for (auto& pkg : packages) {
        unordered_set<int> unique_tokens;
        tokenize(pkg.name + " " + pkg.name + " " + pkg.desc, pkg.tokens, unique_tokens, true);
    }

    vocab.resize(vocab_map.size());
    for (auto& kv : vocab_map) {
        vocab[kv.second.id] = &kv.second;
        vocab[kv.second.id]->word = kv.first;
    }

    for (auto& pkg : packages) {
        unordered_set<int> unique_tokens;
        for (int id : pkg.tokens) unique_tokens.insert(id);
        for (int id : unique_tokens) vocab[id]->doc_freq++;
    }

    int N = packages.size();
    mt19937 rng(42);
    for (auto* w : vocab) {
        w->idf = log((float)N / (1 + w->doc_freq));
        vector<int> indices(D);
        for(int i=0; i<D; ++i) indices[i] = i;
        shuffle(indices.begin(), indices.end(), rng);
        for(int i=0; i<4; ++i) w->pos_idx.push_back(indices[i]);
        for(int i=4; i<8; ++i) w->neg_idx.push_back(indices[i]);
    }

    for (auto& pkg : packages) {
        for (size_t i = 0; i < pkg.tokens.size(); ++i) {
            int w_id = pkg.tokens[i];
            WordInfo* w = vocab[w_id];
            for (size_t j = max(0, (int)i - 3); j < min(pkg.tokens.size(), i + 4); ++j) {
                if (i == j) continue;
                int ctx_id = pkg.tokens[j];
                WordInfo* ctx_w = vocab[ctx_id];
                float weight = ctx_w->idf;
                for(int idx : ctx_w->pos_idx) w->context_vec[idx] += weight;
                for(int idx : ctx_w->neg_idx) w->context_vec[idx] -= weight;
            }
        }
    }

    for (auto* w : vocab) {
        float norm = 0.0f;
        for (int k = 0; k < D; ++k) norm += w->context_vec[k] * w->context_vec[k];
        norm = sqrt(norm);
        if (norm > 0) {
            for (int k = 0; k < D; ++k) w->context_vec[k] /= norm;
        }
    }

    for (auto& pkg : packages) {
        pkg.embedding.resize(D, 0.0f);
        for (int id : pkg.tokens) {
            WordInfo* w = vocab[id];
            float weight = w->idf;
            for (int k = 0; k < D; ++k) {
                pkg.embedding[k] += w->context_vec[k] * weight;
            }
        }
        float norm = 0.0f;
        for (int k = 0; k < D; ++k) norm += pkg.embedding[k] * pkg.embedding[k];
        pkg.norm = sqrt(norm);
        if (pkg.norm > 0) {
            for (int k = 0; k < D; ++k) pkg.embedding[k] /= pkg.norm;
        }
    }
}

float score_exact(const Package& pkg, const vector<string>& q_words) {
    string name = pkg.name;
    string desc = pkg.desc;
    transform(name.begin(), name.end(), name.begin(), ::tolower);
    transform(desc.begin(), desc.end(), desc.begin(), ::tolower);

    float score = 0.0f;
    bool all_in_name = true;
    bool all_in_desc = true;
    for (const string& qw : q_words) {
        if (name.find(qw) == string::npos) all_in_name = false;
        if (desc.find(qw) == string::npos && name.find(qw) == string::npos) all_in_desc = false;
    }
    if (!all_in_desc) return 0.0f;

    if (all_in_name) score += 100.0f;
    else score += 10.0f;

    string joined = q_words[0];
    for(size_t i=1; i<q_words.size(); ++i) joined += " " + q_words[i];
    if (name == joined) score += 1000.0f;
    else if (name.find(joined) != string::npos) score += 50.0f;

    return score;
}

int levenshtein(const string& a, const string& b) {
    int m = a.size(), n = b.size();
    vector<int> dp(n + 1);
    for (int j = 0; j <= n; ++j) dp[j] = j;
    for (int i = 1; i <= m; ++i) {
        int prev = dp[0];
        dp[0] = i;
        for (int j = 1; j <= n; ++j) {
            int temp = dp[j];
            if (a[i - 1] == b[j - 1]) dp[j] = prev;
            else dp[j] = 1 + min({dp[j], dp[j - 1], prev});
            prev = temp;
        }
    }
    return dp[n];
}

unordered_map<int, float> get_fuzzy_words(const string& qw) {
    unordered_map<int, float> res;
    for (auto* w : vocab) {
        if (abs((int)w->word.size() - (int)qw.size()) > 2) continue;
        int dist = levenshtein(qw, w->word);
        if (dist <= 2) {
            float sim = 1.0f - (float)dist / max(qw.size(), w->word.size());
            res[w->id] = sim;
        }
    }
    return res;
}

float score_fuzzy(const Package& pkg, const vector<unordered_map<int, float>>& fuzzy_q) {
    float total = 0.0f;
    for (const auto& fq : fuzzy_q) {
        float best_match = 0.0f;
        for (int id : pkg.tokens) {
            auto it = fq.find(id);
            if (it != fq.end()) {
                float score = it->second * vocab[id]->idf;
                if (score > best_match) best_match = score;
            }
        }
        total += best_match;
    }
    return total;
}

float score_vector(const Package& pkg, const vector<float>& q_emb) {
    if (pkg.norm == 0) return 0.0f;
    float dot = 0.0f;
    for (int k = 0; k < D; ++k) dot += pkg.embedding[k] * q_emb[k];
    return dot;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <query>\n";
        return 1;
    }

    vector<string> q_words;
    for(int i=1; i<argc; ++i) {
        string w = argv[i];
        transform(w.begin(), w.end(), w.begin(), ::tolower);
        string cur = "";
        for (char c : w) {
            if (isalnum(c)) cur += c;
            else if (!cur.empty()) { q_words.push_back(cur); cur = ""; }
        }
        if (!cur.empty()) q_words.push_back(cur);
    }
    if (q_words.empty()) return 0;

    load_packages();
    if (packages.empty()) {
        cerr << "No packages found.\n";
        return 1;
    }

    build_embeddings();

    vector<unordered_map<int, float>> fuzzy_q;
    for (const string& qw : q_words) {
        fuzzy_q.push_back(get_fuzzy_words(qw));
    }

    vector<float> q_emb(D, 0.0f);
    for (const string& qw : q_words) {
        auto it = vocab_map.find(qw);
        if (it != vocab_map.end()) {
            WordInfo* w = vocab[it->second.id];
            for (int k = 0; k < D; ++k) {
                q_emb[k] += w->context_vec[k] * w->idf;
            }
        }
    }
    float q_norm = 0.0f;
    for (int k = 0; k < D; ++k) q_norm += q_emb[k] * q_emb[k];
    q_norm = sqrt(q_norm);
    if (q_norm > 0) {
        for (int k = 0; k < D; ++k) q_emb[k] /= q_norm;
    }

    vector<pair<float, int>> exact_results;
    for (size_t i = 0; i < packages.size(); ++i) {
        float s = score_exact(packages[i], q_words);
        if (s > 0) exact_results.push_back({s, i});
    }
    stable_sort(exact_results.rbegin(), exact_results.rend());

    vector<pair<float, int>> fuzzy_results;
    for (size_t i = 0; i < packages.size(); ++i) {
        float s = score_fuzzy(packages[i], fuzzy_q);
        if (s > 0) fuzzy_results.push_back({s, i});
    }
    stable_sort(fuzzy_results.rbegin(), fuzzy_results.rend());

    vector<pair<float, int>> vector_results;
    for (size_t i = 0; i < packages.size(); ++i) {
        float s = score_vector(packages[i], q_emb);
        if (s > 0) vector_results.push_back({s, i});
    }
    stable_sort(vector_results.rbegin(), vector_results.rend());

    auto get_group_key = [&](const Package& p) {
        string key = p.desc;
        for (char& c : key) if (isdigit(c)) c = '#';
        return key;
    };

    unordered_map<string, vector<int>> desc_to_pkgs;
    for (size_t i = 0; i < packages.size(); ++i) {
        desc_to_pkgs[get_group_key(packages[i])].push_back(i);
    }

    unordered_set<string> seen_desc;
    struct ResRow { string method; float score; int idx; };
    vector<ResRow> final_output;

    auto add_top = [&](const vector<pair<float, int>>& res, const string& method, int limit) {
        int added = 0;
        for (auto& r : res) {
            Package& p = packages[r.second];
            string gk = get_group_key(p);
            if (seen_desc.count(gk)) continue;
            seen_desc.insert(gk);
            final_output.push_back({method, r.first, r.second});
            added++;
            if (added == limit) break;
        }
    };

    add_top(exact_results, "EXACT", 5);
    add_top(fuzzy_results, "FUZZY", 5);
    add_top(vector_results, "VECTOR", 5);

    while (final_output.size() < 15) {
        int before = final_output.size();
        add_top(exact_results, "EXACT", 1);
        if (final_output.size() >= 15) break;
        add_top(fuzzy_results, "FUZZY", 1);
        if (final_output.size() >= 15) break;
        add_top(vector_results, "VECTOR", 1);
        if (final_output.size() == before) break;
    }

    for (auto& fo : final_output) {
        Package& p = packages[fo.idx];
        string also = "";
        for (int other_idx : desc_to_pkgs[get_group_key(p)]) {
            if (other_idx != fo.idx) {
                also += " " + packages[other_idx].name;
            }
        }

        string desc_clean = p.desc;
        replace(desc_clean.begin(), desc_clean.end(), '\n', ' ');
        cout << "[" << fo.method << " " << fo.score << "] " << p.name << " : " << desc_clean;
        if (!also.empty()) cout << " -- ALSO:" << also;
        cout << "\n";
    }

    return 0;
}
