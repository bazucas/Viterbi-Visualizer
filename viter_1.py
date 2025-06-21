import tkinter as tk
from tkinter import font as tkfont, messagebox
import math
from collections import defaultdict, Counter
from functools import reduce
from operator import mul

# ───────────────────────────── UI TRANSLATIONS ────────────────────────────────
TR = {
    'pt': {
        'corpus':     'Corpus (uma frase/palavra-tag por linha):',
        'sentence':   'Frase a analisar:',
        'calculate':  'Calcular',
        'best':       'Probabilidade conjunta:',
        'error_corpus':   'O corpus está vazio ou mal formatado (usa palavra/etiqueta).',
        'error_sentence': 'Introduz uma frase com pelo menos uma palavra.',
        'error_alpha':    'α inválido (usa número ≥ 0).',
        'explanation': 'Explicação',
        'laplace':    'Laplace',
        'alpha':      'α',
        'lang':       '🇵🇹 PT',
    },
    'en': {
        'corpus':     'Corpus (one sent./word-tag per line):',
        'sentence':   'Sentence to analyse:',
        'calculate':  'Calculate',
        'best':       'Joint probability:',
        'error_corpus':   'Corpus is empty or badly formatted (use word/tag).',
        'error_sentence': 'Enter a sentence with at least one word.',
        'error_alpha':    'Invalid α (use number ≥ 0).',
        'explanation': 'Explanation',
        'laplace':    'Laplace',
        'alpha':      'α',
        'lang':       '🇬🇧 EN',
    },
    'fr': {
        'corpus':     'Corpus (une phrase/mot-étiquette par ligne):',
        'sentence':   'Phrase à analyser :',
        'calculate':  'Calculer',
        'best':       'Probabilité conjointe :',
        'error_corpus':   'Le corpus est vide ou mal formé (utilisez mot/étiquette).',
        'error_sentence': 'Saisissez une phrase contenant au moins un mot.',
        'error_alpha':    'α non valide (nombre ≥ 0).',
        'explanation': 'Explication',
        'laplace':    'Laplace',
        'alpha':      'α',
        'lang':       '🇫🇷 FR',
    }
}
FLAG = {'pt': '🇵🇹 PT', 'en': '🇬🇧 EN', 'fr': '🇫🇷 FR'}

# default corpus
default_corpus = (
    'Eu/Pp como/V o/Art gelado/N\n'
    'O/Art almoço/N foi/V bom/Adj\n'
    'Eu/Pp almoço/V todos/Pi os/Art dias/N'
)

# colours per POS tag
TAG_COLOR = {
    'Art': '#b6e9c9',
    'N'  : '#fff5a8',
    'V'  : '#f5b0b0',
    'Adj': '#d0d0ff',
    'Pp' : '#c2e0ff',
    'Pi' : '#ffdfc2',
    '_PATH_': '#ffb347'
}

# ───────────────────────────── HMM TRAINING ───────────────────────────────────
def train_hmm(corpus_lines, alpha: float):
    sentences = []
    for ln in corpus_lines:
        sent = []
        for tok in ln.split():
            if '/' not in tok:
                continue
            w, t = tok.rsplit('/', 1)
            sent.append((w.lower(), t))
        if sent:
            sentences.append(sent)
    if not sentences:
        raise ValueError('empty')

    states = sorted({t for s in sentences for _, t in s})
    S = len(states)

    start_c, trans_c, emit_c = Counter(), defaultdict(Counter), defaultdict(Counter)
    for sent in sentences:
        prev = None
        for i, (w, t) in enumerate(sent):
            emit_c[t][w] += 1
            if i == 0:
                start_c[t] += 1
            if prev is not None:
                trans_c[prev][t] += 1
            prev = t

    # helper
    def lprob(num, den):  # safe log
        return -math.inf if num == 0 else math.log(num / den)

    # start probs
    start_log = {}
    total_start = sum(start_c.values())
    for s in states:
        num = start_c[s] + alpha
        den = total_start + alpha * S
        start_log[s] = lprob(num, den)

    # transition probs
    trans_log = {}
    for s in states:
        total = sum(trans_c[s].values())
        trans_log[s] = {}
        for t in states:
            num = trans_c[s][t] + alpha
            den = total + alpha * S
            trans_log[s][t] = lprob(num, den)

    # emission probs
    emit_log = {}
    for s in states:
        total = sum(emit_c[s].values())
        vocab = set(emit_c[s])
        V = len(vocab)
        emit_log[s] = {}
        # known words
        for w in vocab:
            num = emit_c[s][w] + alpha
            den = total + alpha * (V + 1)   # +1 for <UNK>
            emit_log[s][w] = lprob(num, den)
        # unknown token
        emit_log[s]['<UNK>'] = lprob(alpha, total + alpha * (V + 1))

    return states, start_log, trans_log, emit_log, start_c, trans_c, emit_c

# ───────────────────────────── VITERBI CORE ───────────────────────────────────
def viterbi(words, states, start_log, trans_log, emit_log):
    T = len(words)
    V = [{s: (-math.inf, None) for s in states} for _ in range(T)]

    # initialise
    for s in states:
        V[0][s] = (start_log[s] + emit_log[s].get(words[0], emit_log[s]['<UNK>']), None)

    # recursion
    for t in range(1, T):
        for s in states:
            emis = emit_log[s].get(words[t], emit_log[s]['<UNK>'])
            best_prev, prev_s = max((V[t-1][sp][0] + trans_log[sp][s], sp) for sp in states)
            V[t][s] = (best_prev + emis, prev_s)

    # termination
    last_scores = {s: V[-1][s][0] for s in states}
    best_final = max(last_scores, key=last_scores.get)
    best_log   = last_scores[best_final]

    path = [best_final]
    for t in range(T-1, 0, -1):
        path.append(V[t][path[-1]][1])
    path.reverse()
    return V, path, best_log

# ───────────────────────────── GUI CLASS ──────────────────────────────────────
class ViterbiGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Viterbi visualizador')
        self.geometry('1350x800+80+40')
        self.language = 'pt'

        self.font_ui   = tkfont.Font(size=11)
        self.font_mono = tkfont.Font(family='Courier New', size=11)

        self._build_widgets()
        self._translate_ui()

    # ----------------------------- UI construction ---------------------------
    def _build_widgets(self):
        # top bar: language & Laplace controls
        top = tk.Frame(self); top.pack(fill='x', padx=10, pady=5)

        # language selector
        self.lang_var = tk.StringVar(value=FLAG[self.language])
        tk.OptionMenu(top, self.lang_var, *FLAG.values(),
                      command=self._change_lang).pack(side='right')

        # Laplace checkbox and α entry
        self.laplace_var = tk.BooleanVar(value=False)
        laplace_chk = tk.Checkbutton(top, text=TR['pt']['laplace'],
                                     variable=self.laplace_var,
                                     command=self._laplace_toggle)
        laplace_chk.pack(side='left')

        tk.Label(top, text=TR['pt']['alpha']).pack(side='left', padx=(10, 0))
        self.alpha_var = tk.StringVar(value='1')
        self.ent_alpha = tk.Entry(top, width=4, textvariable=self.alpha_var)
        self.ent_alpha.configure(state='disabled')
        self.ent_alpha.pack(side='left')

        # corpus
        self.lbl_corpus = tk.Label(self, font=self.font_ui)
        self.lbl_corpus.pack(anchor='w', padx=10)

        self.txt_corpus = tk.Text(self, height=6, width=170, font=self.font_mono)
        self.txt_corpus.insert('1.0', default_corpus)
        self.txt_corpus.pack(padx=10, pady=5)

        # sentence
        frame_sent = tk.Frame(self); frame_sent.pack(anchor='w', padx=10, pady=5)
        self.lbl_sentence = tk.Label(frame_sent, font=self.font_ui)
        self.lbl_sentence.grid(row=0, column=0, sticky='e')
        self.ent_sentence = tk.Entry(frame_sent, width=80, font=self.font_ui)
        self.ent_sentence.insert(0, 'Eu almoço o almoço')
        self.ent_sentence.grid(row=0, column=1, padx=(0, 10))
        self.btn_calc = tk.Button(frame_sent, font=self.font_ui,
                                  command=self._calculate)
        self.btn_calc.grid(row=0, column=2)

        # probability table
        self.frm_table = tk.Frame(self)
        self.frm_table.pack(padx=10, pady=(10, 0))

        # best probability
        self.lbl_best = tk.Label(self, font=('Helvetica', 14, 'bold'))
        self.lbl_best.pack(pady=8)

        # explanation
        self.lbl_expl = tk.Label(self, font=('Helvetica', 12, 'bold'))
        self.lbl_expl.pack(anchor='w', padx=10)
        self.txt_expl = tk.Text(self, height=13, width=170,
                                font=self.font_mono, state='disabled')
        self.txt_expl.pack(padx=10, pady=(0, 10))

    # ----------------------------- translations ------------------------------
    def _translate_ui(self):
        t = TR[self.language]
        self.lbl_corpus.config(text=t['corpus'])
        self.lbl_sentence.config(text=t['sentence'])
        self.btn_calc.config(text=t['calculate'])
        self.lbl_best.config(text='')
        self.lbl_expl.config(text=t['explanation'])

    def _change_lang(self, *_):
        for code, flag in FLAG.items():
            if self.lang_var.get() == flag:
                self.language = code
                break
        self._translate_ui()
        if hasattr(self, '_last_result'):
            self._show_explanation(*self._last_result)

    # enable/disable α field
    def _laplace_toggle(self):
        state = 'normal' if self.laplace_var.get() else 'disabled'
        self.ent_alpha.configure(state=state)

    # ----------------------------- main calc ---------------------------------
    def _calculate(self):
        raw_corpus = self.txt_corpus.get('1.0', 'end').strip().split('\n')
        sentence = self.ent_sentence.get().strip()
        if not sentence:
            messagebox.showerror('Error', TR[self.language]['error_sentence'])
            return

        # alpha
        alpha = 0.0
        if self.laplace_var.get():
            try:
                alpha = float(self.alpha_var.get())
                if alpha < 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror('Error', TR[self.language]['error_alpha'])
                return

        try:
            states, s_log, a_log, b_log, s_cnt, a_cnt, b_cnt = train_hmm(raw_corpus, alpha)
        except ValueError:
            messagebox.showerror('Error', TR[self.language]['error_corpus'])
            return

        words = [w.lower() for w in sentence.split()]
        V, path, best_log = viterbi(words, states, s_log, a_log, b_log)

        self._draw_table(words, states, V, path)
        self._show_explanation(words, path, best_log,
                               s_cnt, a_cnt, b_cnt, states, alpha)
        self._last_result = (words, path, best_log,
                             s_cnt, a_cnt, b_cnt, states, alpha)

    # ----------------------------- table -------------------------------------
    def _draw_table(self, words, states, V, path):
        for w in self.frm_table.winfo_children():
            w.destroy()
        tk.Label(self.frm_table, text='', width=10).grid(row=0, column=0)

        for j, w in enumerate(words, 1):
            tk.Label(self.frm_table, text=w, font=self.font_mono,
                     borderwidth=1, relief='ridge',
                     width=10).grid(row=0, column=j, sticky='nsew')

        for i, s in enumerate(states, 1):
            tk.Label(self.frm_table, text=s, font=self.font_mono,
                     bg=TAG_COLOR.get(s, 'white'),
                     borderwidth=1, relief='ridge', width=10).grid(row=i, column=0)
            for j in range(len(words)):
                p = math.exp(V[j][s][0]) if V[j][s][0] > -1e9 else 0.0
                txt = '<0.0001%' if p < 5e-7 else f'{p*100:.4f}%'
                bg = TAG_COLOR['_PATH_'] if s == path[j] else 'white'
                tk.Label(self.frm_table, text=txt, font=self.font_mono,
                         bg=bg, borderwidth=1, relief='ridge',
                         width=10).grid(row=i, column=j+1, sticky='nsew')

    # ----------------------------- explanation -------------------------------
    def _show_explanation(self, words, path, best_log,
                          s_cnt, a_cnt, b_cnt, states, alpha):
        t = TR[self.language]
        self.lbl_best.config(text=f"{t['best']} {math.exp(best_log):.6f}")

        S = len(states)

        # helpers returning (num, den, prob)
        def start_prob(tag):
            num = s_cnt[tag] + alpha
            den = sum(s_cnt.values()) + alpha * S
            return num, den, num / den
        def trans_prob(prev, nxt):
            num = a_cnt[prev][nxt] + alpha
            den = sum(a_cnt[prev].values()) + alpha * S
            return num, den, num / den
        def emit_prob(tag, word):
            vocab = set(b_cnt[tag])
            V = len(vocab)
            num = b_cnt[tag][word] + alpha
            den = sum(b_cnt[tag].values()) + alpha * (V + 1)
            return num, den, num / den

        lines = []
        if self.language == 'pt':
            lines.append(f"Frase analisada: {' '.join(words)}")
            lines.append(f"Sequência mais provável: {' '.join(path)}\n")
            lines.append("Cálculo " + ("com" if alpha > 0 else "sem") + f" Laplace (α={alpha}):")
        elif self.language == 'en':
            lines.append(f"Analysed sentence: {' '.join(words)}")
            lines.append(f"Most probable sequence: {' '.join(path)}\n")
            lines.append("Calculation " + ("with" if alpha > 0 else "without") + f" Laplace (α={alpha}):")
        else:
            lines.append(f"Phrase analysée : {' '.join(words)}")
            lines.append(f"Séquence la plus probable : {' '.join(path)}\n")
            lines.append("Calcul " + ("avec" if alpha > 0 else "sans") + f" Laplace (α={alpha}) :")

        factors = []
        n, d, p = start_prob(path[0]); factors.append(p)
        lines.append(f"π({path[0]}) = ({n}/{d}) = {p:.6f}")
        n, d, p = emit_prob(path[0], words[0]); factors.append(p)
        lines.append(f"b_{path[0]}('{words[0]}') = ({n}/{d}) = {p:.6f}")

        for i in range(1, len(words)):
            prev, curr = path[i-1], path[i]
            n, d, p = trans_prob(prev, curr); factors.append(p)
            lines.append(f"a_{prev}->{curr} = ({n}/{d}) = {p:.6f}")
            n, d, p = emit_prob(curr, words[i]); factors.append(p)
            lines.append(f"b_{curr}('{words[i]}') = ({n}/{d}) = {p:.6f}")

        prod = reduce(mul, factors, 1.0)
        lines.append('-' * 60)
        lines.append("Produto = " + " × ".join(f"{f:.6f}" for f in factors)
                     + f" = {prod:.6f}")

        self.txt_expl.config(state='normal')
        self.txt_expl.delete('1.0', 'end')
        self.txt_expl.insert('end', '\n'.join(lines))
        self.txt_expl.config(state='disabled')

# ────────────────────────────── MAIN ─────────────────────────────────────────-
if __name__ == '__main__':
    ViterbiGUI().mainloop()
