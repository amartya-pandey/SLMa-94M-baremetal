#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#if defined _WIN32
#include "win.h"
#else
#include <unistd.h>
#include <sys/mman.h>
#endif

// n_layer is same as num_hidden_layers
// in python i have used num_hidden_layers due to some problem with llama.cpp when i was trying to use it to run this model on mobile termux

typedef struct
{
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
} Config;

typedef struct
{
    float *token_embedding_table;
    float *rms_att_weight;
    float *rms_ffn_weight;
    float *wq;
    float *wk;
    float *wv;
    float *wo;
    float *w1;
    float *w2;
    float *w3;
    float *rms_final_weight;
    float *wcls;
} TransformerWeights;

typedef struct
{
    float *x;
    float *xb;
    float *xb2;
    float *hb;
    float *hb2;
    float *q;
    float *k;
    float *v;
    float *att;
    float *logits;
    float *key_cache;
    float *value_cache;
} RunState;

void malloc_run_state(RunState *s, Config *p)
{
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->q = calloc(p->dim, sizeof(float));
    s->k = calloc(kv_dim, sizeof(float));
    s->v = calloc(kv_dim, sizeof(float));
    s->key_cache = calloc((size_t)p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc((size_t)p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(abs(p->vocab_size), sizeof(float));

    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 ||
        !s->q || !s->k || !s->v || !s->key_cache || !s->value_cache ||
        !s->att || !s->logits)
    {
        fprintf(stderr, "Fatal: memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState *s)
{
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

typedef struct
{
    float *rms_att;
    float *rms_ffn;
    float *wq;
    float *wk;
    float *wv;
    float *wo;
    float *w1;
    float *w2;
    float *w3;
} LayerWeights;

typedef struct
{
    float *token_embedding_table;
    float *rms_final_weight;
    float *wcls;
    LayerWeights *layer;
} TWeights;

typedef struct
{
    Config config;
    TWeights weights;
    RunState state;
    int fd;
    float *data;
    ssize_t file_size;
} Transformer2;

typedef Transformer2 Transformer;

void memory_map_weights(TWeights *w, Config *p, float *ptr)
{
    int vocab_size = abs(p->vocab_size);
    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    int kv_dim = (dim * p->n_kv_heads) / p->n_heads;

    w->token_embedding_table = ptr;
    ptr += (size_t)vocab_size * dim;

    w->layer = malloc(p->n_layers * sizeof(LayerWeights));
    if (!w->layer)
    {
        fprintf(stderr, "Fatal: layer weight table alloc failed\n");
        exit(1);
    }

    for (int l = 0; l < p->n_layers; l++)
    {
        w->layer[l].rms_att = ptr;
        ptr += dim;
        w->layer[l].rms_ffn = ptr;
        ptr += dim;
        w->layer[l].wq = ptr;
        ptr += (size_t)dim * dim;
        w->layer[l].wk = ptr;
        ptr += (size_t)dim * kv_dim;
        w->layer[l].wv = ptr;
        ptr += (size_t)dim * kv_dim;
        w->layer[l].wo = ptr;
        ptr += (size_t)dim * dim;
        w->layer[l].w1 = ptr;
        ptr += (size_t)dim * hidden_dim;
        w->layer[l].w2 = ptr;
        ptr += (size_t)dim * hidden_dim;
        w->layer[l].w3 = ptr;
        ptr += (size_t)hidden_dim * dim;
    }

    w->rms_final_weight = ptr;
    w->wcls = w->token_embedding_table;
}

void read_checkpoint(char *checkpoint, Config *config, TWeights *weights,
                     int *fd, float **data, ssize_t *file_size)
{
    FILE *file = fopen(checkpoint, "rb");
    if (!file)
    {
        fprintf(stderr, "Cannot open checkpoint: %s\n", checkpoint);
        exit(EXIT_FAILURE);
    }
    if (fread(config, sizeof(Config), 1, file) != 1)
    {
        fprintf(stderr, "Failed to read header\n");
        exit(EXIT_FAILURE);
    }

    int shared_weights = config->vocab_size < 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);

    fseek(file, 0, SEEK_END);
    *file_size = ftell(file);
    fclose(file);

    *fd = open(checkpoint, O_RDONLY);
    if (*fd == -1)
    {
        fprintf(stderr, "open() failed\n");
        exit(EXIT_FAILURE);
    }
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED)
    {
        fprintf(stderr, "mmap() failed\n");
        exit(EXIT_FAILURE);
    }

    float *weights_ptr = *data + sizeof(Config) / sizeof(float);
    memory_map_weights(weights, config, weights_ptr);
    (void)shared_weights;
}

void build_transformer(Transformer *t, char *checkpoint_path)
{
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer2 *t)
{
    if (t->data != MAP_FAILED)
        munmap(t->data, t->file_size);
    if (t->fd != -1)
        close(t->fd);
    free(t->weights.layer);
    free_run_state(&t->state);
}

void rmsnorm(float *o, float *x, float *weight, int size)
{
    float ss = 0.0f;
    for (int j = 0; j < size; j++)
        ss += x[j] * x[j];
    ss = 1.0f / sqrtf(ss / size + 1e-6f);
    for (int j = 0; j < size; j++)
        o[j] = weight[j] * (ss * x[j]);
}

void softmax(float *x, int size)
{
    float max_val = x[0];
    for (int i = 1; i < size; i++)
        if (x[i] > max_val)
            max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++)
    {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++)
        x[i] /= sum;
}

void matmul(float *xout, float *x, float *w, int n, int d)
{
    int i;
#pragma omp parallel for private(i)
    for (i = 0; i < d; i++)
    {
        float val = 0.0f;
        for (int j = 0; j < n; j++)
            val += w[i * n + j] * x[j];
        xout[i] = val;
    }
}

static inline void apply_rope_head(float *vec, int head_size, int pos)
{
    int half = head_size / 2;
    for (int i = 0; i < half; i++)
    {
        float freq = 1.0f / powf(10000.0f, (2.0f * i) / head_size);
        float angle = pos * freq;
        float c = cosf(angle), s = sinf(angle);
        float v0 = vec[i];
        float v1 = vec[i + half];
        vec[i] = v0 * c - v1 * s;
        vec[i + half] = v1 * c + v0 * s;
    }
}

static void apply_rope(float *q, float *k, int n_heads, int n_kv_heads, int head_size, int pos)
{
    for (int h = 0; h < n_heads; h++)
        apply_rope_head(q + h * head_size, head_size, pos);
    for (int h = 0; h < n_kv_heads; h++)
        apply_rope_head(k + h * head_size, head_size, pos);
}

float *forward(Transformer *transformer, int token, int pos)
{
    Config *p = &transformer->config;
    TWeights *w = &transformer->weights;
    RunState *s = &transformer->state;
    int dim = p->dim;
    int kv_dim = (dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;
    float *x = s->x;
    memcpy(x, w->token_embedding_table + token * dim, dim * sizeof(float));

    for (int l = 0; l < p->n_layers; l++)
    {
        LayerWeights *lw = &w->layer[l];
        rmsnorm(s->xb, x, lw->rms_att, dim);

        matmul(s->q, s->xb, lw->wq, dim, dim);
        matmul(s->k, s->xb, lw->wk, dim, kv_dim);
        matmul(s->v, s->xb, lw->wv, dim, kv_dim);
        apply_rope(s->q, s->k, p->n_heads, p->n_kv_heads, head_size, pos);

        int loff = l * p->seq_len * kv_dim;
        memcpy(s->key_cache + loff + pos * kv_dim, s->k, kv_dim * sizeof(float));
        memcpy(s->value_cache + loff + pos * kv_dim, s->v, kv_dim * sizeof(float));

        int h;
#pragma omp parallel for private(h)
        for (h = 0; h < p->n_heads; h++)
        {
            float *q = s->q + h * head_size;
            float *att = s->att + h * p->seq_len;
            int kv_head = h / kv_mul;

            for (int t = 0; t <= pos; t++)
            {
                float *k = s->key_cache + loff + t * kv_dim + kv_head * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++)
                    score += q[i] * k[i];
                att[t] = score / sqrtf((float)head_size);
            }
            softmax(att, pos + 1);

            float *xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++)
            {
                float *v = s->value_cache + loff + t * kv_dim + kv_head * head_size;
                float a = att[t];
                for (int i = 0; i < head_size; i++)
                    xb[i] += a * v[i];
            }
        }

        matmul(s->xb2, s->xb, lw->wo, dim, dim);
        for (int i = 0; i < dim; i++)
            x[i] += s->xb2[i];

        rmsnorm(s->xb, x, lw->rms_ffn, dim);

        matmul(s->hb, s->xb, lw->w1, dim, hidden_dim);
        matmul(s->hb2, s->xb, lw->w2, dim, hidden_dim);
        for (int i = 0; i < hidden_dim; i++)
        {
            float val = s->hb[i];
            val *= 1.0f / (1.0f + expf(-val));
            val *= s->hb2[i];
            s->hb[i] = val;
        }
        matmul(s->xb, s->hb, lw->w3, hidden_dim, dim);

        for (int i = 0; i < dim; i++)
            x[i] += s->xb[i];
    }

    rmsnorm(x, x, w->rms_final_weight, dim);
    matmul(s->logits, x, w->wcls, dim, abs(p->vocab_size));
    return s->logits;
}

typedef struct
{
    char *str;
    int id;
} TokenIndex;

typedef struct
{
    int bos_id, eos_id, unk_id, byte_offset;
    char **vocab;
    float *vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];

} Tokenizer;

int compare_tokens(const void *a, const void *b)
{
    return strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str);
}

void build_tokenizer(Tokenizer *t, char *tokenizer_path, int vocab_size)
{
    t->vocab_size = vocab_size;
    t->vocab = malloc(vocab_size * sizeof(char *));
    t->vocab_scores = malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL;

    for (int i = 0; i < 256; i++)
    {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }

    FILE *file = fopen(tokenizer_path, "rb");
    if (!file)
    {
        fprintf(stderr, "Cannot open tokenizer: %s\n", tokenizer_path);
        exit(EXIT_FAILURE);
    }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1 ||
        fread(&t->vocab_size, sizeof(int), 1, file) != 1 ||
        fread(&t->bos_id, sizeof(int), 1, file) != 1 ||
        fread(&t->eos_id, sizeof(int), 1, file) != 1 ||
        fread(&t->unk_id, sizeof(int), 1, file) != 1)
    {
        fprintf(stderr, "Tokenizer header read failed\n");
        exit(EXIT_FAILURE);
    }

    int len;
    for (int i = 0; i < vocab_size; i++)
    {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1)
        {
            exit(EXIT_FAILURE);
        }
        if (fread(&len, sizeof(int), 1, file) != 1)
        {
            exit(EXIT_FAILURE);
        }
        t->vocab[i] = (char *)malloc(len + 1);
        if (len > 0 && fread(t->vocab[i], 1, len, file) != len)
        {
            fprintf(stderr, "Failed reading token string at index %d\n", i);
            exit(EXIT_FAILURE);
        }
        t->vocab[i][len] = '\0';
    }
    fclose(file);
    t->byte_offset = 4;
}

void free_tokenizer(Tokenizer *t)
{
    for (int i = 0; i < t->vocab_size; i++)
        free(t->vocab[i]);
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char *decode(Tokenizer *t, int prev_token, int token)
{
    char *piece = t->vocab[token];

    if (token == t->bos_id)
        return "";
    if (token == t->eos_id)
        return "";
    if (token == t->unk_id)
        return "🗑";

    if (prev_token == t->bos_id && piece[0] == ' ')
        piece++;

    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1)
        piece = (char *)t->byte_pieces + byte_val * 2;
    return piece;
}

void safe_printf(char *piece)
{
    if (piece == NULL || piece[0] == '\0')
        return;
    if (piece[1] == '\0')
    {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val)))
            return;
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size)
{
    TokenIndex tok = {.str = str};
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens)
{
    if (text == NULL)
    {
        fprintf(stderr, "Encode error: null text\n");
        exit(EXIT_FAILURE);
    }

    if (t->sorted_vocab == NULL)
    {
        t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++)
        {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    char *str_buffer = malloc((t->max_token_length * 2 + 3) * sizeof(char));
    size_t str_len = 0;
    *n_tokens = 0;

    if (bos && t->bos_id >= 0)
        tokens[(*n_tokens)++] = t->bos_id;

    if (text[0] != '\0')
    {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        if (dummy_prefix != -1)
            tokens[(*n_tokens)++] = dummy_prefix;
    }

    for (char *c = text; *c != '\0'; c++)
    {
        if ((*c & 0xC0) != 0x80)
            str_len = 0;
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';
        if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4)
            continue;

        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        if (id != -1)
        {
            tokens[(*n_tokens)++] = id;
        }
        else
        {

            for (int i = 0; i < (int)str_len; i++)
            {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0;
    }

    while (1)
    {
        float best_score = -1e10f;
        int best_id = -1;
        int best_idx = -1;

        for (int i = 0; i < (*n_tokens - 1); i++)
        {
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score)
            {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }
        if (best_idx == -1)
            break;

        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++)
            tokens[i] = tokens[i + 1];
        (*n_tokens)--;
    }

    for (int i = 0; i < (int)str_len; i++)
    {
        tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + t->byte_offset;
    }
    if (eos && t->eos_id >= 0)
        tokens[(*n_tokens)++] = t->eos_id;
    free(str_buffer);
}

typedef struct
{
    float prob;
    int index;
} ProbIndex;

typedef struct
{
    int vocab_size;
    ProbIndex *probindex;
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float *probs, int n)
{
    int max_i = 0;
    for (int i = 1; i < n; i++)
        if (probs[i] > probs[max_i])
            max_i = i;
    return max_i;
}

int sample_mult(float *probs, int n, float coin)
{
    float cdf = 0.0f;
    for (int i = 0; i < n; i++)
    {
        cdf += probs[i];
        if (coin < cdf)
            return i;
    }
    return n - 1;
}

int compare_prob(const void *a, const void *b)
{
    float diff = ((ProbIndex *)b)->prob - ((ProbIndex *)a)->prob;
    return (diff > 0) - (diff < 0);
}

int sample_topp(float *probs, int n, float topp, ProbIndex *probindex, float coin)
{
    const float cutoff = (1.0f - topp) / (n - 1);
    int n0 = 0;
    for (int i = 0; i < n; i++)
    {
        if (probs[i] >= cutoff)
        {
            probindex[n0].index = i;
            probindex[n0].prob = probs[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare_prob);

    float cumul = 0.0f;
    int last_idx = n0 - 1;
    for (int i = 0; i < n0; i++)
    {
        cumul += probindex[i].prob;
        if (cumul > topp)
        {
            last_idx = i;
            break;
        }
    }

    float r = coin * cumul, cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++)
    {
        cdf += probindex[i].prob;
        if (r < cdf)
            return probindex[i].index;
    }
    return probindex[last_idx].index;
}

void build_sampler(Sampler *s, int vocab_size, float temperature, float topp, unsigned long long seed)
{
    s->vocab_size = vocab_size;
    s->temperature = temperature;
    s->topp = topp;
    s->rng_state = seed;
    s->probindex = malloc(vocab_size * sizeof(ProbIndex));
}
void free_sampler(Sampler *s) { free(s->probindex); }

unsigned int random_u32(unsigned long long *state)
{
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { return (random_u32(state) >> 8) / 16777216.0f; }

int sample(Sampler *sampler, float *logits)
{
    if (sampler->temperature == 0.0f)
        return sample_argmax(logits, sampler->vocab_size);
    for (int q = 0; q < sampler->vocab_size; q++)
        logits[q] /= sampler->temperature;
    softmax(logits, sampler->vocab_size);
    float coin = random_f32(&sampler->rng_state);
    if (sampler->topp <= 0 || sampler->topp >= 1)
        return sample_mult(logits, sampler->vocab_size, coin);
    return sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
}

long time_in_ms()
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
              char *prompt, int steps)
{
    char *empty_prompt = "";
    if (prompt == NULL)
        prompt = empty_prompt;

    int num_prompt_tokens = 0;
    int *prompt_tokens = (int *)malloc((strlen(prompt) + 3) * sizeof(int));
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1)
    {
        fprintf(stderr, "Encode produced no tokens\n");
        exit(EXIT_FAILURE);
    }

    long start = 0;
    int next, token = prompt_tokens[0], pos = 0;

    while (pos < steps)
    {
        float *logits = forward(transformer, token, pos);

        if (pos < num_prompt_tokens - 1)
        {
            next = prompt_tokens[pos + 1];
        }
        else
        {
            next = sample(sampler, logits);
        }
        pos++;

        if (next == tokenizer->bos_id || next == tokenizer->eos_id)
            break;

        char *piece = decode(tokenizer, token, next);
        safe_printf(piece);
        fflush(stdout);
        token = next;

        if (start == 0)
            start = time_in_ms();
    }
    printf("\n");
    if (pos > 1)
    {
        long end = time_in_ms();
        fprintf(stderr, "Tokens/sec: %.2f\n", (pos - 1) / (double)(end - start) * 1000.0);
    }
    free(prompt_tokens);
}

void read_stdin(const char *guide, char *buffer, size_t bufsize)
{
    printf("%s", guide);
    fflush(stdout);
    if (fgets(buffer, bufsize, stdin) != NULL)
    {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n')
            buffer[len - 1] = '\0';
    }
}

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
          char *cli_user_prompt, char *cli_system_prompt, int steps)
{
    char system_prompt[512];
    char user_prompt[512];

    char rendered_prompt[1152];

    int num_prompt_tokens = 0;
    int *prompt_tokens = (int *)malloc(1152 * sizeof(int));
    int user_idx;
    int8_t user_turn = 1;
    int next = 0, token = 0, pos = 0;

    while (pos < steps)
    {
        if (user_turn)
        {
            if (pos == 0)
            {
                if (cli_system_prompt == NULL)
                    read_stdin("System prompt (optional, press Enter to skip): ",
                               system_prompt, sizeof(system_prompt));
                else
                    strncpy(system_prompt, cli_system_prompt, sizeof(system_prompt) - 1);
            }

            if (pos == 0 && cli_user_prompt != NULL)
                strncpy(user_prompt, cli_user_prompt, sizeof(user_prompt) - 1);
            else
                read_stdin("User: ", user_prompt, sizeof(user_prompt));

            if (pos == 0 && system_prompt[0] != '\0')
                snprintf(rendered_prompt, sizeof(rendered_prompt),
                         "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]",
                         system_prompt, user_prompt);
            else
                snprintf(rendered_prompt, sizeof(rendered_prompt),
                         "[INST] %s [/INST]", user_prompt);

            encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
            user_idx = 0;
            user_turn = 0;
            printf("Assistant: ");
            fflush(stdout);
        }

        token = (user_idx < num_prompt_tokens) ? prompt_tokens[user_idx++] : next;
        if (token == tokenizer->eos_id)
        {
            user_turn = 1;
        }

        float *logits = forward(transformer, token, pos);
        next = sample(sampler, logits);
        pos++;

        if (user_idx >= num_prompt_tokens && next != tokenizer->eos_id)
        {
            char *piece = decode(tokenizer, token, next);
            safe_printf(piece);
            fflush(stdout);
        }
        if (next == tokenizer->eos_id)
            printf("\n");
        if (pos >= steps)
        {
            printf("\n[Max context reached]\n");
            break;
        }
    }
    free(prompt_tokens);
}

#ifndef TESTING
void error_usage()
{
    fprintf(stderr, "Usage:   ./run <model.bin> [options]\n");
    fprintf(stderr, "Example: ./run model.bin -z tokenizer.bin -i \"Hello, \"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature [0, inf]  default 0.7\n");
    fprintf(stderr, "  -p <float>  top-p value [0, 1]    default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed            default time(NULL)\n");
    fprintf(stderr, "  -n <int>    steps to generate     default 256  (0 = seq_len)\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> tokenizer binary path  default tokenizer.bin\n");
    fprintf(stderr, "  -m <string> mode: generate|chat    default generate\n");
    fprintf(stderr, "  -y <string> system prompt (chat mode)\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[])
{
    char *checkpoint_path = NULL;
    char *tokenizer_path = "tokenizer.bin";
    float temperature = 0.7f;
    float topp = 0.9f;
    int steps = 256;
    char *prompt = NULL;
    unsigned long long rng_seed = 0;
    char *mode = "generate";
    char *system_prompt = NULL;

    if (argc < 2)
        error_usage();
    checkpoint_path = argv[1];

    for (int i = 2; i < argc; i += 2)
    {
        if (i + 1 >= argc || argv[i][0] != '-' || strlen(argv[i]) != 2)
            error_usage();
        if (argv[i][1] == 't')
            temperature = atof(argv[i + 1]);
        else if (argv[i][1] == 'p')
            topp = atof(argv[i + 1]);
        else if (argv[i][1] == 's')
            rng_seed = (unsigned long long)atoll(argv[i + 1]);
        else if (argv[i][1] == 'n')
            steps = atoi(argv[i + 1]);
        else if (argv[i][1] == 'i')
            prompt = argv[i + 1];
        else if (argv[i][1] == 'z')
            tokenizer_path = argv[i + 1];
        else if (argv[i][1] == 'm')
            mode = argv[i + 1];
        else if (argv[i][1] == 'y')
            system_prompt = argv[i + 1];
        else
            error_usage();
    }

    if (rng_seed == 0)
        rng_seed = (unsigned long long)time(NULL);
    if (temperature < 0.0f)
        temperature = 0.0f;
    if (topp < 0.0f || topp > 1.0f)
        topp = 0.9f;
    if (steps < 0)
        steps = 0;

    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len)
        steps = transformer.config.seq_len;

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    if (strcmp(mode, "generate") == 0)
        generate(&transformer, &tokenizer, &sampler, prompt, steps);
    else if (strcmp(mode, "chat") == 0)
        chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
    else
        error_usage();

    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
#endif
