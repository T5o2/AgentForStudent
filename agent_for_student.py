"""
AgentForStudent - prototype implementation
Runs on students_by_term_150.xlsx (sheets: grade log + student summary).

Steps:
  1. Load, rename, validate and clean the data
  2. Feature engineering (term GPA, domains, prerequisites)
  3. Bayesian Knowledge Tracing per skill domain
  4. Academic-risk prediction (Logistic Regression vs Random Forest)
  5. Priority scheduling (CSP with greedy + constraint checks) and SM-2
  6. Retrieval step of RAG (TF-IDF + cosine similarity)
  7. ReAct-style agent loop that uses all of the above as tools
"""
import sys, io, itertools, datetime as dt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA = sys.argv[1] if len(sys.argv) > 1 else "students_by_term_150.xlsx"
OUT = sys.argv[2] if len(sys.argv) > 2 else "out"
import os; os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.axisbelow": True, "figure.dpi": 150})

LOG = io.StringIO()
def say(*a):
    s = " ".join(str(x) for x in a)
    print(s); LOG.write(s + "\n")

# ---------------------------------------------------------------- 1. LOAD + CLEAN
grades = pd.read_excel(DATA, sheet_name=0)
summary = pd.read_excel(DATA, sheet_name=1)
raw_sample = grades.head(4).copy()

grades.columns = ["student_id", "name", "level", "term", "course_code", "course_name",
                  "credits", "score", "letter", "points", "cgpa"]
summary.columns = ["student_id", "name", "level", "cgpa", "n_courses", "total_credits"]

grades["level_no"] = grades["level"].str.extract(r"(\d+)").astype(int)
grades["term_no"] = grades["term"].str.extract(r"(\d+)").astype(int)
summary["level_no"] = summary["level"].str.extract(r"(\d+)").astype(int)

GRADE_SCALE = [(95, "A+", 4.0), (90, "A", 3.75), (85, "B+", 3.5), (80, "B", 3.0),
               (75, "C+", 2.5), (70, "C", 2.0), (65, "D+", 1.5), (60, "D", 1.0), (0, "F", 0.0)]
def to_letter(s):
    for lo, l, p in GRADE_SCALE:
        if s >= lo: return l, p

checks = {}
checks["rows (grade log)"] = len(grades)
checks["unique students"] = grades.student_id.nunique()
checks["unique courses"] = grades.course_code.nunique()
checks["missing values"] = int(grades.isna().sum().sum() + summary.isna().sum().sum())
checks["duplicate (student, course) rows"] = int(grades.duplicated(["student_id", "course_code"]).sum())
checks["scores outside 0-100"] = int((~grades.score.between(0, 100)).sum())
exp = grades.score.apply(to_letter)
checks["letter/score mismatches"] = int((exp.str[0] != grades.letter).sum())
checks["points/letter mismatches"] = int((exp.str[1] != grades.points).sum())
checks["duplicate student names (use ID as key)"] = int(summary.name.duplicated().sum())

gp = grades[grades.credits > 0].assign(qp=lambda d: d.credits * d.points)
calc = gp.groupby("student_id").agg(qp=("qp", "sum"), cr=("credits", "sum"))
calc["cgpa_calc"] = (calc.qp / calc.cr).round(2)
chk = summary.set_index("student_id").join(calc)
checks["CGPA recomputation mismatches"] = int((abs(chk.cgpa - chk.cgpa_calc) > 0.01).sum())
checks["F grades (failed, no retake row)"] = int((grades.letter == "F").sum())
checks["zero-credit rows (CUR1101)"] = int((grades.credits == 0).sum())
checks["max credits in plan"] = int(grades.groupby("student_id").credits.sum().max())

# cleaning actions
grades["name"] = grades["name"].str.strip()
grades["passed"] = (grades.letter != "F").astype(int)
# earned credits must exclude failed courses
earned = grades[grades.passed == 1].groupby("student_id").credits.sum().rename("earned_credits")
summary = summary.merge(earned, on="student_id", how="left")

# ---------------------------------------------------------------- 2. FEATURES
DOMAIN = {}
for c in ["MTH1141", "MTH1441", "MTH1602", "MTH2144", "CS1302", "PHYS1113"]: DOMAIN[c] = "Math"
for c in ["CS1011", "CS1012", "CS2015", "CS2107", "SE1201", "HCI2011"]: DOMAIN[c] = "Programming"
for c in ["CEN2011", "CEN2012", "CS3608", "CEN3011", "SEC1101", "DS2201"]: DOMAIN[c] = "Systems"
for c in grades.course_code.unique():
    if c.startswith("AI"): DOMAIN[c] = "AI"
for c in grades.course_code.unique():
    DOMAIN.setdefault(c, "General")
grades["domain"] = grades.course_code.map(DOMAIN)

PREREQ = {"CS1012": ["CS1011"], "CS2015": ["CS1012"], "CS2107": ["CS2015", "CS1302"],
          "MTH2144": ["MTH1141"], "CEN2012": ["CEN2011"], "CS3608": ["CEN2012"],
          "CEN3011": ["CS3608"], "AI3101": ["AI2001", "CS2015"],
          "AI3201": ["MTH1441", "MTH1602", "CS2015"], "AI3301": ["MTH1602"],
          "AI3102": ["AI3101"], "AI3202": ["AI3201"], "AI3302": ["AI3202"],
          "AI3104": ["AI3101"], "AI4203": ["AI3202"], "AI4303": ["AI3202"],
          "SEC1101": ["CEN3011"], "ELIE1202": ["ELIE1201"]}

term_gpa = (gp.groupby(["student_id", "term_no"])
              .apply(lambda d: (d.credits * d.points).sum() / d.credits.sum(), include_groups=False)
              .round(2).rename("term_gpa").reset_index())
grades = grades.merge(term_gpa, on=["student_id", "term_no"], how="left")

def history_features(g):
    """features for each course row using ONLY terms before it (no leakage)"""
    rows = []
    g = g.sort_values("term_no")
    tg = g.drop_duplicates("term_no").set_index("term_no").term_gpa
    for _, r in g[g.term_no >= 2].iterrows():
        prev = g[g.term_no < r.term_no]
        pc = prev[prev.credits > 0]
        prev_gpa = (pc.credits * pc.points).sum() / pc.credits.sum()
        dom = prev[prev.domain == r.domain].score
        pre = prev[prev.course_code.isin(PREREQ.get(r.course_code, []))].score
        terms = tg[tg.index < r.term_no]
        rows.append(dict(student_id=r.student_id, course_code=r.course_code,
                         term_no=r.term_no, domain=r.domain, credits=r.credits,
                         prev_cgpa=round(prev_gpa, 2), last_term_gpa=terms.iloc[-1],
                         gpa_trend=round(terms.iloc[-1] - terms.iloc[0], 2),
                         domain_prior_avg=round(dom.mean() if len(dom) else prev.score.mean(), 1),
                         prereq_avg=round(pre.mean() if len(pre) else prev.score.mean(), 1),
                         has_prereq=int(len(pre) > 0), min_prior_score=prev.score.min(),
                         prior_fails=int((prev.letter == "F").sum()),
                         score=r.score, at_risk=int(r.score < 75)))
    return rows

feat = pd.DataFrame(list(itertools.chain.from_iterable(
    history_features(g) for _, g in grades.groupby("student_id"))))
feat = pd.concat([feat, pd.get_dummies(feat.domain, prefix="dom", dtype=int)], axis=1)

# ---------------------------------------------------------------- 3. BKT
def bkt_trace(obs, p0, pT, pG, pS):
    """returns P(L) after each observation and the log-likelihood"""
    pL, out, ll = p0, [], 0.0
    for o in obs:
        p_correct = pL * (1 - pS) + (1 - pL) * pG
        ll += np.log(p_correct if o else 1 - p_correct)
        if o: post = pL * (1 - pS) / p_correct
        else: post = pL * pS / (1 - p_correct)
        pL = post + (1 - post) * pT
        out.append(pL)
    return out, ll

grades["correct"] = (grades.score >= 80).astype(int)   # mastery evidence = B or better
seqs = {d: [g.sort_values("term_no").correct.tolist() for _, g in grp.groupby("student_id")]
        for d, grp in grades[grades.domain != "General"].groupby("domain")}
grid = dict(p0=[.1, .2, .3, .4, .5, .6, .7, .8], pT=[.02, .05, .1, .15, .2, .3], pG=[.1, .2, .3], pS=[.05, .1, .15, .2])
BKT_PARAMS = {}
for d, ss in seqs.items():
    best = max(itertools.product(*grid.values()),
               key=lambda p: sum(bkt_trace(s, *p)[1] for s in ss))
    BKT_PARAMS[d] = dict(zip(grid.keys(), best))

def mastery(student_id):
    g = grades[(grades.student_id == student_id) & (grades.domain != "General")].sort_values("term_no")
    res = {}
    for d, gd in g.groupby("domain"):
        res[d] = round(float(bkt_trace(gd.correct.tolist(), *BKT_PARAMS[d].values())[0][-1]), 2)
    return res

# ---------------------------------------------------------------- 4. RISK MODEL
FEATS = ["prev_cgpa", "last_term_gpa", "gpa_trend", "domain_prior_avg", "prereq_avg",
         "has_prereq", "min_prior_score", "prior_fails", "credits", "term_no"] + \
        [c for c in feat.columns if c.startswith("dom_")]
X, y, groups = feat[FEATS], feat.at_risk, feat.student_id
tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X, y, groups))
models = {"Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
          "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                  class_weight="balanced", random_state=42, n_jobs=-1)}
metrics = {}
for n, m in models.items():
    m.fit(X.iloc[tr], y.iloc[tr])
    p = m.predict_proba(X.iloc[te])[:, 1]; yh = (p >= .5).astype(int)
    metrics[n] = dict(Accuracy=accuracy_score(y.iloc[te], yh), Precision=precision_score(y.iloc[te], yh),
                      Recall=recall_score(y.iloc[te], yh), F1=f1_score(y.iloc[te], yh),
                      ROC_AUC=roc_auc_score(y.iloc[te], p))
rf = models["Random Forest"]
cm = confusion_matrix(y.iloc[te], (rf.predict_proba(X.iloc[te])[:, 1] >= .5).astype(int))
imp = pd.Series(rf.feature_importances_, index=FEATS).sort_values()

def predict_risk(student_id, upcoming_courses):
    """build features for courses the student has NOT taken yet"""
    g = grades[grades.student_id == student_id]
    nxt = g.term_no.max() + 1
    fake = pd.concat([g, pd.DataFrame([dict(student_id=student_id, course_code=c, term_no=nxt,
                      domain=DOMAIN[c], credits=3, score=0, letter="?", points=0, term_gpa=np.nan)
                      for c in upcoming_courses])])
    f = pd.DataFrame(history_features(fake))
    f = f[f.term_no == nxt]
    for c in FEATS:
        if c.startswith("dom_"): f[c] = (f.domain == c[4:]).astype(int)
    return dict(zip(f.course_code, rf.predict_proba(f[FEATS])[:, 1].round(2)))

# ---------------------------------------------------------------- 5. SCHEDULING + SM-2
W_U, W_W, W_D = 3.0, 4.0, 2.0
def priority(days_left, weight, mastery_score):
    return W_U / (days_left + 1) + W_W * weight + W_D * (1 - mastery_score)

def sm2(quality_seq, ef=2.5):
    reps, interval, out = 0, 0, []
    for q in quality_seq:
        if q < 3: reps, interval = 0, 1
        else:
            reps += 1
            interval = 1 if reps == 1 else 6 if reps == 2 else round(interval * ef)
        ef = max(1.3, ef + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02)))
        out.append((q, round(ef, 2), interval))
    return out

def build_schedule(tasks, start, days=7, max_per_day=5):
    """greedy CSP: highest priority first, each hour slot checked against constraints"""
    lectures = {0: range(8, 13), 1: range(10, 15), 2: range(8, 13), 3: range(10, 15), 4: range(8, 12)}
    study_hours = range(14, 22)                      # 2pm-10pm; sleep window never used
    cal = {}
    load = {d: 0 for d in range(days)}
    for t in sorted(tasks, key=lambda t: -t["priority"]):
        need = t["hours"]
        last = min(t["days_left"], days) - 1             # must finish before deadline
        for d in range(last, -1, -1):                     # latest feasible day first keeps early days free for urgent tasks
            if need == 0: break
            day = start + dt.timedelta(days=d)
            wd = (day.weekday() + 1) % 7                    # Sunday = 0
            run = 0
            for h in study_hours:
                if need == 0 or load[d] >= max_per_day or run >= 2: break
                if h in lectures.get(wd, []) or (d, h) in cal: continue
                cal[(d, h)] = t["course"] + " " + t["task"]
                load[d] += 1; need -= 1; run += 1
        t["unscheduled"] = need
    return cal

# ---------------------------------------------------------------- 6. RAG retrieval step
CORPUS = [
 ("AI3201 L4", "Overfitting happens when a model learns noise in the training data so training accuracy is high but test accuracy is low. Use cross validation regularization and more data."),
 ("AI3201 L3", "Gradient descent updates the model weights in the opposite direction of the gradient of the loss function using a learning rate."),
 ("AI3201 L5", "Bias variance trade off: simple models underfit with high bias while complex models overfit with high variance."),
 ("AI3201 L6", "Decision trees split data using information gain or Gini impurity. Random forests average many trees trained on bootstrap samples."),
 ("AI3101 L2", "Uninformed search: breadth first search explores level by level and depth first search explores the deepest node first."),
 ("AI3101 L4", "A* search uses f(n) = g(n) + h(n) and is optimal when the heuristic is admissible."),
 ("AI3101 L6", "Constraint satisfaction problems use variables domains and constraints and are solved with backtracking and forward checking."),
 ("SEC1101 L1", "The CIA triad means confidentiality integrity and availability, the three goals of information security."),
 ("SEC1101 L3", "Symmetric encryption uses one shared key while public key encryption uses a public and a private key."),
 ("AI3301 L2", "Data cleaning handles missing values duplicates and outliers before analysis and visualization."),
]
vec = TfidfVectorizer(stop_words="english")
doc_m = vec.fit_transform([t for _, t in CORPUS])
def retrieve(query, k=3):
    s = cosine_similarity(vec.transform([query]), doc_m)[0]
    return [(CORPUS[i][0], round(float(s[i]), 3), CORPUS[i][1]) for i in s.argsort()[::-1][:k]]

# ---------------------------------------------------------------- 7. ReAct LOOP
UPCOMING_T5 = ["AI3201", "AI3101", "SEC1101", "AI3301", "HCI2011"]
lvl4 = summary[summary.level_no == 4].student_id
risk_all = {s: predict_risk(s, UPCOMING_T5) for s in lvl4}
demo_id = min(risk_all, key=lambda s: abs(risk_all[s]["AI3201"] - 0.65))
TODAY = dt.date(2026, 10, 11)   # Sunday
TASKS = [dict(course="AI3201", task="Quiz 2", days_left=3, weight=.05, hours=3),
         dict(course="AI3301", task="Lab report", days_left=2, weight=.05, hours=2),
         dict(course="AI3101", task="Assignment 1", days_left=5, weight=.10, hours=4),
         dict(course="SEC1101", task="Midterm", days_left=6, weight=.25, hours=5),
         dict(course="AI3201", task="Midterm", days_left=9, weight=.25, hours=5),
         dict(course="HCI2011", task="Project milestone", days_left=12, weight=.15, hours=3)]

trace = []
def T(kind, text): trace.append((kind, text))
srow = summary[summary.student_id == demo_id].iloc[0]
T("Thought", "New week started. Check the student's academic state first.")
T("Action", f"get_transcript(student_id={demo_id})")
lt = grades[grades.student_id == demo_id].groupby("term_no").term_gpa.first()
T("Observation", f"Level {srow.level_no}, CGPA {srow.cgpa}, term GPAs {[float(x) for x in lt.values]}, earned credits {srow.earned_credits}")
T("Action", f"bkt_mastery(student_id={demo_id})")
m = mastery(demo_id); T("Observation", str(m))
T("Thought", "Next term includes AI3201 and AI3101. Predict risk for each upcoming course.")
T("Action", f"predict_risk(student_id={demo_id}, courses={UPCOMING_T5})")
rk = {k: float(v) for k, v in risk_all[demo_id].items()}; T("Observation", str(rk))
for t in TASKS:
    dom_m = m.get(DOMAIN[t["course"]], .5)
    t["mastery"] = round(0.5 * dom_m + 0.5 * (1 - rk[t["course"]]), 2)
    t["priority"] = round(priority(t["days_left"], t["weight"], t["mastery"]), 2)
    t["hours"] = int(round(t["hours"] * (1 + 0.5 * (1 - t["mastery"]))))
T("Thought", "Risk for AI3201 is high and Quiz 2 is in 3 days. Rank deadlines with the priority score, then fill free slots.")
T("Action", "get_calendar(week=2026-10-11) ; build_schedule(tasks)")
cal = build_schedule(TASKS, TODAY)
T("Observation", f"{len(cal)} study hours placed over 7 days, unscheduled hours = {sum(t['unscheduled'] for t in TASKS)}")
T("Thought", "Student asked: why is my model good on training data but bad on test data? Ground the answer in course material.")
T("Action", 'retrieve("model good on training data but bad on test data")')
rr = retrieve("model good on training data but bad on test data")
T("Observation", "; ".join(f"{a} (cos={b})" for a, b, _ in rr))
T("Action", "schedule_review(topic='AI3201 Overfitting', algorithm='SM-2')")
sm = sm2([3, 4, 5]); T("Observation", f"review after {sm[0][2]} day, then {sm[1][2]} days, then {sm[2][2]} days")
T("Final", f"Send plan + 3 study links to student {demo_id}; notify advisor because predicted risk in AI3201 is {rk['AI3201']}.")

# ---------------------------------------------------------------- OUTPUTS
def term_png(lines, path, title, width=11):
    h = 0.24 * len(lines) + 0.7
    fig = plt.figure(figsize=(width, h)); fig.patch.set_facecolor("#1e1e1e")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_facecolor("#1e1e1e")
    ax.text(0.012, 1 - 0.18 / h, "●  ●  ●   " + title, color="#9a9a9a", family="DejaVu Sans Mono", fontsize=9, va="top", transform=ax.transAxes)
    for i, (txt, col) in enumerate(lines):
        ax.text(0.012, 1 - (0.55 + 0.24 * i) / h, txt, color=col, family="DejaVu Sans Mono", fontsize=8.6, va="top", transform=ax.transAxes)
    fig.savefig(path, dpi=170, facecolor=fig.get_facecolor()); plt.close(fig)

W, G, Y, C, Gr = "#e6e6e6", "#7ee787", "#f2cc60", "#79c0ff", "#9a9a9a"
L = [("$ python agent_for_student.py students_by_term_150.xlsx", G), ("[1] Data validation report", Y)]
L += [(f"    {k:<42}: {v}", W) for k, v in checks.items()]
L += [("[2] Cleaning actions", Y), ("    renamed 11 Arabic columns to English snake_case; extracted level_no, term_no", W),
      ("    added passed flag; earned_credits now excludes F grades", W),
      ("    added domain, term_gpa, correct(score>=80) and leakage-free history features", W),
      (f"    feature table: {feat.shape[0]} rows x {len(FEATS)} model features, at_risk rate = {feat.at_risk.mean():.1%}", G)]
term_png(L, f"{OUT}/shot_1_validation.png", "terminal - data validation")

L = [("$ python agent_for_student.py --stage models", G), ("[3] BKT parameters fitted by grid search (max log-likelihood)", Y)]
L += [(f"    {d:<12} p(L0)={p['p0']:.2f}  p(T)={p['pT']:.2f}  p(G)={p['pG']:.2f}  p(S)={p['pS']:.2f}", W) for d, p in BKT_PARAMS.items()]
L += [("[4] Risk prediction (target: score < 75), split by student 80/20", Y),
      (f"    train rows={len(tr)}  test rows={len(te)}  test students={groups.iloc[te].nunique()}", Gr),
      (f"    {'model':<22}{'Acc':>7}{'Prec':>7}{'Rec':>7}{'F1':>7}{'AUC':>7}", C)]
L += [(f"    {n:<22}" + "".join(f"{v:>7.2f}" for v in mm.values()), W) for n, mm in metrics.items()]
term_png(L, f"{OUT}/shot_2_models.png", "terminal - model training")

colors = {"Thought": C, "Action": Y, "Observation": W, "Final": G}
L = [(f"$ python agent_for_student.py --agent --student {demo_id}", G)]
import textwrap
for k, v in trace:
    for j, ln in enumerate(textwrap.wrap(v, 96)):
        L.append(((f"{k + ':':<13}" if j == 0 else " " * 13) + ln, colors[k]))
term_png(L, f"{OUT}/shot_3_react.png", "terminal - ReAct agent loop", width=11.5)

# schedule table figure
days = [TODAY + dt.timedelta(days=d) for d in range(7)]
fig, ax = plt.subplots(figsize=(10, 4.4))
ax.set_xlim(0, 7); ax.set_ylim(22, 8); ax.grid(False)
cmap = {"AI3201": BLUE, "AI3301": AQUA, "AI3101": VIOLET, "SEC1101": ORANGE, "HCI2011": MAGENTA}
lect = {0: range(8, 13), 1: range(10, 15), 2: range(8, 13), 3: range(10, 15), 4: range(8, 12)}
for d, day in enumerate(days):
    wd = (day.weekday() + 1) % 7
    for h in lect.get(wd, []):
        ax.add_patch(plt.Rectangle((d + .04, h + .04), .92, .92, color="#d9d8d3"))
    for h in range(8, 22):
        if (d, h) in cal:
            txt = cal[(d, h)]
            ax.add_patch(plt.Rectangle((d + .04, h + .04), .92, .92, color=cmap[txt.split()[0]]))
            ax.text(d + .5, h + .52, txt.replace(" ", "\n", 1), ha="center", va="center", fontsize=6.3, color="white")
ax.set_xticks([d + .5 for d in range(7)], [d.strftime("%a %d") for d in days])
ax.set_yticks(range(8, 23, 2), [f"{h}:00" for h in range(8, 23, 2)])
ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)
ax.set_title(f"Generated weekly study plan for student {demo_id} (grey = lectures)", loc="left", fontsize=11, color=INK)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_schedule.png"); plt.close(fig)

# BKT trajectory for demo student
fig, ax = plt.subplots(figsize=(7.5, 3.6))
g = grades[(grades.student_id == demo_id) & (grades.domain != "General")].sort_values(["term_no"])
for d, col in zip(["Math", "Programming", "Systems", "AI"], [BLUE, ORANGE, AQUA, VIOLET]):
    gd = g[g.domain == d]
    if gd.empty: continue
    tr_ = bkt_trace(gd.correct.tolist(), *BKT_PARAMS[d].values())[0]
    ax.plot(range(1, len(tr_) + 1), tr_, color=col, lw=2, marker="o", ms=5, label=d)
    ax.annotate(f"{d} {tr_[-1]:.2f}", (len(tr_), tr_[-1]), xytext=(6, 0), textcoords="offset points", fontsize=8, color=INK, va="center")
ax.axhline(0.6, color=RED, lw=1, ls="--"); ax.text(0.9, 0.62, "remediation threshold 0.60", fontsize=8, color=MUTED)
ax.set_ylim(0, 1); ax.set_xlabel("Observation (course, in term order)"); ax.set_ylabel("P(mastered)")
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=8)
ax.set_title(f"BKT mastery trajectory by domain, student {demo_id}", loc="left", fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_bkt.png"); plt.close(fig)

# model results: feature importance + confusion matrix
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw=dict(width_ratios=[1.6, 1]))
top = imp.tail(8)
a1.barh(top.index, top.values, color=BLUE, height=.6); a1.grid(axis="y", visible=False)
a1.set_title("Random Forest feature importance", loc="left", fontsize=11); a1.set_xlabel("importance")
a2.imshow(cm, cmap="Blues"); a2.grid(False)
for (i, j), v in np.ndenumerate(cm):
    a2.text(j, i, v, ha="center", va="center", color="white" if v > cm.max() / 2 else INK, fontsize=12)
a2.set_xticks([0, 1], ["pred: OK", "pred: at risk"]); a2.set_yticks([0, 1], ["true: OK", "true: at risk"])
a2.set_title("Confusion matrix (test set)", loc="left", fontsize=11)
for s in a2.spines.values(): s.set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_model.png"); plt.close(fig)

# dataset overview: letter distribution + mean term GPA
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4))
order = ["A+", "A", "B+", "B", "C+", "C", "D+", "D", "F"]
vc = grades.letter.value_counts().reindex(order)
a1.bar(order, vc.values, color=[BLUE] * 8 + [RED], width=.65); a1.grid(axis="x", visible=False)
a1.set_title("Letter-grade distribution (7,912 records)", loc="left", fontsize=11)
for i, v in enumerate(vc.values): a1.text(i, v + 30, f"{v:,}", ha="center", fontsize=7.5, color=MUTED)
mt = term_gpa.groupby("term_no").term_gpa.mean()
a2.plot(mt.index, mt.values, color=BLUE, lw=2, marker="o", ms=5); a2.set_ylim(2, 4)
a2.set_xlabel("Term"); a2.set_ylabel("mean term GPA"); a2.set_title("Average term GPA by term", loc="left", fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_overview.png"); plt.close(fig)

# ---------------------------------------------------------------- save tables for the report
sched_rows = sorted(TASKS, key=lambda t: -t["priority"])
with pd.ExcelWriter(f"{OUT}/students_clean_features.xlsx") as xw:
    grades.drop(columns=["correct"]).to_excel(xw, sheet_name="grades_clean", index=False)
    summary.to_excel(xw, sheet_name="student_summary", index=False)
    feat.to_excel(xw, sheet_name="model_features", index=False)
    pd.DataFrame([(c, DOMAIN[c], ", ".join(PREREQ.get(c, []))) for c in sorted(DOMAIN)],
                 columns=["course_code", "domain", "prerequisites"]).to_excel(xw, sheet_name="courses", index=False)
    pd.DataFrame(checks.items(), columns=["check", "result"]).to_excel(xw, sheet_name="validation", index=False)

import json
json.dump(dict(checks=checks, metrics=metrics, bkt=BKT_PARAMS, demo_id=int(demo_id),
               mastery=m, risk={k: float(v) for k, v in rk.items()},
               tasks=sched_rows, sm2=sm, retrieve=rr, n_feat_rows=len(feat),
               at_risk_rate=float(feat.at_risk.mean()), train=len(tr), test=len(te),
               test_students=int(groups.iloc[te].nunique()), importance=imp.round(3).to_dict(),
               cm=cm.tolist(), raw_sample=raw_sample.astype(str).values.tolist(),
               raw_cols=list(raw_sample.columns),
               clean_sample=grades.loc[raw_sample.index, ["student_id", "level_no", "term_no", "course_code", "credits", "score", "letter", "points", "domain", "term_gpa", "passed"]].astype(str).values.tolist(),
               feat_sample=feat[feat.student_id == demo_id].head(4)[["course_code", "term_no", "prev_cgpa", "last_term_gpa", "gpa_trend", "domain_prior_avg", "prereq_avg", "at_risk"]].astype(str).values.tolist(),
               summary_sample=summary[summary.student_id == demo_id][["student_id", "level_no", "cgpa", "n_courses", "total_credits", "earned_credits"]].astype(str).values.tolist(),
               fail_example=summary[summary.total_credits != summary.earned_credits].head(1)[["student_id", "total_credits", "earned_credits"]].astype(str).values.tolist(),
               ), open(f"{OUT}/results.json", "w"), indent=1, default=str)
say("done", demo_id)
