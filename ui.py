import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
import threading
import numpy as np
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

GROUPS = {
    'A': ['Mexico', 'South Africa', 'South Korea', 'Czechia'],
    'B': ['Canada', 'Bosnia and Herzegovina', 'Qatar', 'Switzerland'],
    'C': ['Brazil', 'Haiti', 'Morocco', 'Scotland'],
    'D': ['Australia', 'Paraguay', 'Turkey', 'United States'],
    'E': ['Curaçao', 'Ecuador', 'Germany', 'Ivory Coast'],
    'F': ['Japan', 'Netherlands', 'Sweden', 'Tunisia'],
    'G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    'H': ['Cape Verde', 'Saudi Arabia', 'Spain', 'Uruguay'],
    'I': ['France', 'Iraq', 'Norway', 'Senegal'],
    'J': ['Algeria', 'Argentina', 'Austria', 'Jordan'],
    'K': ['Colombia', 'DR Congo', 'Portugal', 'Uzbekistan'],
    'L': ['Croatia', 'England', 'Ghana', 'Panama'],
}

ALL_TEAMS = sorted(set(t for g in GROUPS.values() for t in g))

# ── colours ──────────────────────────────────────────────────────────────────
BG        = '#1a1a2e'
CARD      = '#16213e'
ACCENT    = '#0f3460'
GREEN     = '#1D9E75'
BLUE      = '#378ADD'
GREY      = '#888888'
WHITE     = '#e0e0e0'
DIM       = '#666680'
RED       = '#e24b4a'
GOLD      = '#f5a623'

# ── simulation ────────────────────────────────────────────────────────────────
def run_simulation(actual_results):
    model        = joblib.load(os.path.join(DATA_DIR, 'model.pkl'))
    feature_cols = joblib.load(os.path.join(DATA_DIR, 'feature_cols.pkl'))
    features     = pd.read_csv(os.path.join(DATA_DIR, 'features.csv'),
                                parse_dates=['date'])

    wc2026 = features[
        (features['date'] >= '2026-06-11') &
        (features['is_world_cup'] == 1)
    ].copy()

    probs = model.predict_proba(wc2026[feature_cols])
    wc2026['p_away_win'] = probs[:, 0]
    wc2026['p_draw']     = probs[:, 1]
    wc2026['p_home_win'] = probs[:, 2]

    match_probs = {}
    for _, row in wc2026.iterrows():
        match_probs[(row['home_team'], row['away_team'])] = (
            row['p_home_win'], row['p_draw'], row['p_away_win']
        )

    completed = {}
    for r in actual_results:
        completed[(r['home_team'], r['away_team'])] = \
            (r['home_score'], r['away_score'])

    N = 10000
    qualify = {t: 0 for g in GROUPS.values() for t in g}
    winners = {t: 0 for g in GROUPS.values() for t in g}
    pts_tot = {t: 0 for g in GROUPS.values() for t in g}

    for _ in range(N):
        thirds = []
        for grp, teams in GROUPS.items():
            pts = {t: 0 for t in teams}
            gd  = {t: 0 for t in teams}
            gf  = {t: 0 for t in teams}

            for i, ta in enumerate(teams):
                for tb in teams[i+1:]:

                    if (ta, tb) in completed:
                        hs, as_ = completed[(ta, tb)]
                    elif (tb, ta) in completed:
                        as_, hs = completed[(tb, ta)]
                    else:
                        hs = as_ = None

                    if hs is not None:
                        if hs > as_:   pts[ta] += 3
                        elif hs == as_: pts[ta] += 1; pts[tb] += 1
                        else:           pts[tb] += 3
                        gd[ta] += hs - as_; gd[tb] += as_ - hs
                        gf[ta] += hs;       gf[tb] += as_
                        continue

                    if (ta, tb) in match_probs:
                        pa, pd_, pb = match_probs[(ta, tb)]
                    elif (tb, ta) in match_probs:
                        pb, pd_, pa = match_probs[(tb, ta)]
                    else:
                        pa, pd_, pb = 0.4, 0.2, 0.4

                    tot = pa + pd_ + pb
                    pa /= tot; pd_ /= tot; pb /= tot

                    oc = np.random.choice(['a','d','b'], p=[pa, pd_, pb])

                    if oc == 'a':
                        ag = np.random.choice([1,2,3], p=[.45,.35,.20])
                        bg = np.random.choice([0,1],   p=[.65,.35])
                        pts[ta] += 3
                    elif oc == 'd':
                        ag = bg = np.random.choice([0,1,2], p=[.30,.45,.25])
                        pts[ta] += 1; pts[tb] += 1
                    else:
                        bg = np.random.choice([1,2,3], p=[.45,.35,.20])
                        ag = np.random.choice([0,1],   p=[.65,.35])
                        pts[tb] += 3

                    gd[ta] += ag-bg; gd[tb] += bg-ag
                    gf[ta] += ag;    gf[tb] += bg

            st = sorted(teams, key=lambda t:(pts[t],gd[t],gf[t]), reverse=True)
            qualify[st[0]] += 1
            qualify[st[1]] += 1
            winners[st[0]] += 1
            for t in teams:
                pts_tot[t] += pts[t]
            thirds.append((pts[st[2]], gd[st[2]], gf[st[2]], st[2]))

        thirds.sort(reverse=True)
        for _,_,_,t in thirds[:8]:
            qualify[t] += 1

    return [
        {
            'group': g, 'team': t,
            'qualify': round(qualify[t]/N*100, 1),
            'win_group': round(winners[t]/N*100, 1),
            'avg_pts': round(pts_tot[t]/N, 1),
        }
        for g, teams in GROUPS.items() for t in teams
    ]


def load_results():
    path = os.path.join(DATA_DIR, 'actual_results.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def save_results(results):
    path = os.path.join(DATA_DIR, 'actual_results.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)


# ── app ───────────────────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('World Cup 2026 Predictor')
        self.configure(bg=BG)
        self.geometry('1100x750')
        self.minsize(900, 600)
        self.resizable(True, True)

        self.actual_results = load_results()
        self.sim_data       = []
        self._build_ui()
        self._refresh(first=True)

    # ── layout ────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # header
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill='x', padx=24, pady=(20, 0))
        tk.Label(hdr, text='World Cup 2026', font=('Arial', 20, 'bold'),
                 bg=BG, fg=WHITE).pack(side='left')
        self.status_lbl = tk.Label(hdr, text='Simulating…',
                                   font=('Arial', 11), bg=BG, fg=DIM)
        self.status_lbl.pack(side='right', pady=6)

        # tab bar
        tab_bar = tk.Frame(self, bg=BG)
        tab_bar.pack(fill='x', padx=24, pady=(12, 0))
        self.tab_btns = {}
        for name in ('Groups', 'Ranking', 'Enter results', 'Accuracy'):
            b = tk.Button(tab_bar, text=name, font=('Arial', 11),
                          bg=ACCENT, fg=WHITE, relief='flat',
                          padx=14, pady=5, cursor='hand2',
                          command=lambda n=name: self._show_tab(n))
            b.pack(side='left', padx=(0, 6))
            self.tab_btns[name] = b

        # pages
        self.pages = {}
        container = tk.Frame(self, bg=BG)
        container.pack(fill='both', expand=True, padx=24, pady=16)

        for name in ('Groups', 'Ranking', 'Enter results', 'Accuracy'):
            page = tk.Frame(container, bg=BG)
            page.place(relwidth=1, relheight=1)
            self.pages[name] = page

        self._build_groups_page()
        self._build_ranking_page()
        self._build_results_page()
        self._build_accuracy_page()
        self._show_tab('Groups')

    def _show_tab(self, name):
        for n, b in self.tab_btns.items():
            b.configure(bg=WHITE if n == name else ACCENT,
                        fg=BG     if n == name else WHITE)
        for n, p in self.pages.items():
            if n == name:
                p.lift()
            else:
                p.lower()

    # ── groups page ──────────────────────────────────────────────────────────
    def _build_groups_page(self):
        page = self.pages['Groups']
        canvas = tk.Canvas(page, bg=BG, highlightthickness=0)
        sb = ttk.Scrollbar(page, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)
        self.groups_frame = tk.Frame(canvas, bg=BG)
        self.groups_win = canvas.create_window((0,0), window=self.groups_frame,
                                                anchor='nw')
        self.groups_frame.bind('<Configure>',
            lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>',
            lambda e: canvas.itemconfig(self.groups_win, width=e.width))
        canvas.bind_all('<MouseWheel>',
            lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), 'units'))

    def _render_groups(self):
        for w in self.groups_frame.winfo_children():
            w.destroy()

        by_group = {}
        for row in self.sim_data:
            by_group.setdefault(row['group'], []).append(row)

        cols = 3
        for idx, (g, teams) in enumerate(sorted(by_group.items())):
            teams.sort(key=lambda r: -r['qualify'])
            card = tk.Frame(self.groups_frame, bg=CARD,
                            padx=14, pady=12, relief='flat')
            r, c = divmod(idx, cols)
            card.grid(row=r, column=c, padx=8, pady=8, sticky='nsew')
            self.groups_frame.columnconfigure(c, weight=1)

            tk.Label(card, text=f'GROUP {g}', font=('Arial', 10, 'bold'),
                     bg=CARD, fg=DIM).pack(anchor='w', pady=(0,8))

            for i, row in enumerate(teams):
                pct = row['qualify']
                color = GREEN if pct >= 75 else BLUE if pct >= 50 else GREY

                row_f = tk.Frame(card, bg=CARD)
                row_f.pack(fill='x', pady=3)

                # medal
                medal = ['🥇','🥈','·','·'][i]
                tk.Label(row_f, text=medal, font=('Arial', 11),
                         bg=CARD, fg=WHITE, width=2).pack(side='left')

                tk.Label(row_f, text=row['team'], font=('Arial', 11),
                         bg=CARD, fg=WHITE, width=18, anchor='w').pack(side='left')

                # bar
                bar_bg = tk.Frame(row_f, bg=ACCENT, height=6, width=120)
                bar_bg.pack(side='left', padx=6)
                bar_bg.pack_propagate(False)
                fill_w = max(2, int(pct / 100 * 120))
                tk.Frame(bar_bg, bg=color, height=6,
                         width=fill_w).place(x=0, y=0)

                tk.Label(row_f, text=f'{pct}%', font=('Arial', 10),
                         bg=CARD, fg=color, width=6).pack(side='left')

                pts_lbl = f'{row["avg_pts"]}pts'
                tk.Label(row_f, text=pts_lbl, font=('Arial', 9),
                         bg=CARD, fg=DIM).pack(side='right')

    # ── ranking page ─────────────────────────────────────────────────────────
    def _build_ranking_page(self):
        page = self.pages['Ranking']
        cols = ('Rank', 'Team', 'Group', 'Qualify %', 'Win Group %', 'Avg Pts')
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Custom.Treeview',
            background=CARD, foreground=WHITE,
            fieldbackground=CARD, rowheight=28,
            borderwidth=0, font=('Arial', 11))
        style.configure('Custom.Treeview.Heading',
            background=ACCENT, foreground=WHITE,
            font=('Arial', 11, 'bold'), borderwidth=0)
        style.map('Custom.Treeview', background=[('selected', ACCENT)])

        self.tree = ttk.Treeview(page, columns=cols, show='headings',
                                  style='Custom.Treeview')
        widths = [50, 200, 70, 100, 120, 90]
        for col, w in zip(cols, widths):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=w, anchor='center')
        self.tree.column('Team', anchor='w')

        sb = ttk.Scrollbar(page, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        self.tree.pack(fill='both', expand=True)

    def _render_ranking(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        sorted_data = sorted(self.sim_data, key=lambda r: -r['qualify'])
        for i, row in enumerate(sorted_data, 1):
            self.tree.insert('', 'end', values=(
                i, row['team'], row['group'],
                f"{row['qualify']}%",
                f"{row['win_group']}%",
                row['avg_pts']
            ))

    # ── enter results page ───────────────────────────────────────────────────
    def _build_results_page(self):
        page = self.pages['Enter results']

        form = tk.Frame(page, bg=CARD, padx=16, pady=14)
        form.pack(fill='x', pady=(0, 16))
        tk.Label(form, text='Add a result', font=('Arial', 13, 'bold'),
                 bg=CARD, fg=WHITE).grid(row=0, column=0, columnspan=7,
                                          sticky='w', pady=(0,10))

        self.home_var  = tk.StringVar()
        self.away_var  = tk.StringVar()
        self.home_score_var = tk.IntVar(value=0)
        self.away_score_var = tk.IntVar(value=0)

        tk.Label(form, text='Home team', bg=CARD, fg=DIM,
                 font=('Arial',10)).grid(row=1, column=0, padx=(0,4))
        ttk.Combobox(form, textvariable=self.home_var,
                     values=ALL_TEAMS, width=22,
                     state='readonly').grid(row=1, column=1, padx=4)

        tk.Label(form, text='Score', bg=CARD, fg=DIM,
                 font=('Arial',10)).grid(row=1, column=2, padx=4)
        tk.Spinbox(form, from_=0, to=20, textvariable=self.home_score_var,
                   width=4, bg=ACCENT, fg=WHITE,
                   buttonbackground=ACCENT).grid(row=1, column=3, padx=2)
        tk.Label(form, text='–', bg=CARD, fg=WHITE,
                 font=('Arial',13)).grid(row=1, column=4)
        tk.Spinbox(form, from_=0, to=20, textvariable=self.away_score_var,
                   width=4, bg=ACCENT, fg=WHITE,
                   buttonbackground=ACCENT).grid(row=1, column=5, padx=2)

        tk.Label(form, text='Away team', bg=CARD, fg=DIM,
                 font=('Arial',10)).grid(row=1, column=6, padx=4)
        ttk.Combobox(form, textvariable=self.away_var,
                     values=ALL_TEAMS, width=22,
                     state='readonly').grid(row=1, column=7, padx=4)

        tk.Button(form, text='Add result', bg=GREEN, fg=WHITE,
                  font=('Arial',11,'bold'), relief='flat', padx=12, pady=4,
                  cursor='hand2',
                  command=self._add_result).grid(row=1, column=8, padx=(12,0))

        # results list
        tk.Label(page, text='Results entered', font=('Arial', 12, 'bold'),
                 bg=BG, fg=WHITE).pack(anchor='w', pady=(0,6))

        list_frame = tk.Frame(page, bg=BG)
        list_frame.pack(fill='both', expand=True)
        canvas = tk.Canvas(list_frame, bg=BG, highlightthickness=0)
        sb2 = ttk.Scrollbar(list_frame, orient='vertical',
                             command=canvas.yview)
        canvas.configure(yscrollcommand=sb2.set)
        sb2.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)
        self.results_inner = tk.Frame(canvas, bg=BG)
        self.results_win = canvas.create_window((0,0),
                            window=self.results_inner, anchor='nw')
        self.results_inner.bind('<Configure>',
            lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>',
            lambda e: canvas.itemconfig(self.results_win, width=e.width))

    def _render_results_list(self):
        for w in self.results_inner.winfo_children():
            w.destroy()

        if not self.actual_results:
            tk.Label(self.results_inner, text='No results added yet.',
                     bg=BG, fg=DIM, font=('Arial',11)).pack(pady=12)
            return

        for i, r in enumerate(self.actual_results):
            row_f = tk.Frame(self.results_inner, bg=CARD, padx=12, pady=8)
            row_f.pack(fill='x', pady=3)

            tk.Label(row_f,
                text=f"{r['home_team']}  {r['home_score']} – {r['away_score']}  {r['away_team']}",
                font=('Arial', 11), bg=CARD, fg=WHITE).pack(side='left')

            tk.Button(row_f, text='✕', bg=CARD, fg=RED,
                      font=('Arial',11), relief='flat', cursor='hand2',
                      command=lambda idx=i: self._delete_result(idx)
                      ).pack(side='right')

    def _add_result(self):
        ht  = self.home_var.get()
        at  = self.away_var.get()
        hs  = self.home_score_var.get()
        as_ = self.away_score_var.get()
        if not ht or not at or ht == at:
            messagebox.showerror('Error', 'Please select two different teams.')
            return
        self.actual_results.append({
            'home_team': ht, 'away_team': at,
            'home_score': hs, 'away_score': as_
        })
        save_results(self.actual_results)
        self._refresh()

    def _delete_result(self, idx):
        self.actual_results.pop(idx)
        save_results(self.actual_results)
        self._refresh()

    # ── accuracy page ─────────────────────────────────────────────────────────
    def _build_accuracy_page(self):
        page = self.pages['Accuracy']

        self.acc_header = tk.Label(page, text='No results yet.',
                                   font=('Arial', 13, 'bold'),
                                   bg=BG, fg=WHITE)
        self.acc_header.pack(anchor='w', pady=(0, 12))

        canvas = tk.Canvas(page, bg=BG, highlightthickness=0)
        sb = ttk.Scrollbar(page, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)
        self.acc_inner = tk.Frame(canvas, bg=BG)
        acc_win = canvas.create_window((0,0), window=self.acc_inner, anchor='nw')
        self.acc_inner.bind('<Configure>',
            lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>',
            lambda e: canvas.itemconfig(acc_win, width=e.width))

    def _render_accuracy(self):
        for w in self.acc_inner.winfo_children():
            w.destroy()

        if not self.actual_results or not self.sim_data:
            self.acc_header.configure(text='No results yet.')
            return

        model        = joblib.load(os.path.join(DATA_DIR, 'model.pkl'))
        feature_cols = joblib.load(os.path.join(DATA_DIR, 'feature_cols.pkl'))
        features     = pd.read_csv(os.path.join(DATA_DIR, 'features.csv'),
                                    parse_dates=['date'])
        wc2026 = features[
            (features['date'] >= '2026-06-11') &
            (features['is_world_cup'] == 1)
        ].copy()
        probs = model.predict_proba(wc2026[feature_cols])
        wc2026['p_away_win'] = probs[:, 0]
        wc2026['p_draw']     = probs[:, 1]
        wc2026['p_home_win'] = probs[:, 2]

        match_lookup = {}
        for _, row in wc2026.iterrows():
            match_lookup[(row['home_team'], row['away_team'])] = row

        labels = {0: 'Away win', 1: 'Draw', 2: 'Home win'}
        correct = 0

        for r in self.actual_results:
            ht, at = r['home_team'], r['away_team']
            hs, as_ = r['home_score'], r['away_score']

            if (ht, at) in match_lookup:
                mrow = match_lookup[(ht, at)]
                ph, pd_, pa = mrow['p_home_win'], mrow['p_draw'], mrow['p_away_win']
            elif (at, ht) in match_lookup:
                mrow = match_lookup[(at, ht)]
                ph, pd_, pa = mrow['p_away_win'], mrow['p_draw'], mrow['p_home_win']
            else:
                continue

            predicted = 2 if ph > pd_ and ph > pa else 1 if pd_ > pa else 0
            actual    = 2 if hs > as_ else 1 if hs == as_ else 0
            is_correct = predicted == actual
            if is_correct:
                correct += 1

            row_f = tk.Frame(self.acc_inner, bg=CARD, padx=12, pady=8)
            row_f.pack(fill='x', pady=3)

            tk.Label(row_f,
                text=f'{ht}  {hs}–{as_}  {at}',
                font=('Arial', 11), bg=CARD, fg=WHITE,
                width=40, anchor='w').pack(side='left')
            tk.Label(row_f,
                text=f'Predicted: {labels[predicted]}',
                font=('Arial', 10), bg=CARD, fg=DIM,
                width=18).pack(side='left')
            tk.Label(row_f,
                text='✓ Correct' if is_correct else '✗ Wrong',
                font=('Arial', 10, 'bold'),
                bg=CARD,
                fg=GREEN if is_correct else RED).pack(side='right', padx=8)

        total = len(self.actual_results)
        pct   = round(correct / total * 100, 1) if total else 0
        self.acc_header.configure(
            text=f'Accuracy: {correct}/{total} correct  ({pct}%)')

    # ── refresh ───────────────────────────────────────────────────────────────
    def _refresh(self, first=False):
        self.status_lbl.configure(text='Simulating…', fg=GOLD)
        self.update()

        def worker():
            data = run_simulation(self.actual_results)
            self.after(0, lambda: self._apply(data))

        threading.Thread(target=worker, daemon=True).start()

    def _apply(self, data):
        self.sim_data = data
        self._render_groups()
        self._render_ranking()
        self._render_results_list()
        self._render_accuracy()
        n = len(self.actual_results)
        self.status_lbl.configure(
            text=f'{n} result(s) entered · simulation updated',
            fg=GREEN)


if __name__ == '__main__':
    App().mainloop()