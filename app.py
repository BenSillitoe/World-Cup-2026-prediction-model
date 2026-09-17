from flask import Flask, jsonify, render_template_string
import pandas as pd
import numpy as np
import os
import joblib
import json

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

def run_simulation():
    model        = joblib.load(os.path.join(DATA_DIR, 'model.pkl'))
    feature_cols = joblib.load(os.path.join(DATA_DIR, 'feature_cols.pkl'))
    features     = pd.read_csv(os.path.join(DATA_DIR, 'features.csv'), parse_dates=['date'])

    wc2026 = features[
        (features['date'] >= '2026-06-11') &
        (features['is_world_cup'] == 1)
    ].copy()

    groups = {
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

    # --- Load actual results played so far ---
    actual_results_path = os.path.join(DATA_DIR, 'actual_results.json')
    actual_results = []
    if os.path.exists(actual_results_path):
        with open(actual_results_path) as f:
            actual_results = json.load(f)

    # Build a set of completed matches and their scores
    completed = {}
    for r in actual_results:
        completed[(r['home_team'], r['away_team'])] = (r['home_score'], r['away_score'])

    # --- Get model probabilities for all fixtures ---
    X_pred       = wc2026[feature_cols]
    probabilities = model.predict_proba(X_pred)

    wc2026['p_away_win'] = probabilities[:, 0]
    wc2026['p_draw']     = probabilities[:, 1]
    wc2026['p_home_win'] = probabilities[:, 2]

    match_probs = {}
    for _, row in wc2026.iterrows():
        match_probs[(row['home_team'], row['away_team'])] = (
            row['p_home_win'], row['p_draw'], row['p_away_win']
        )

    # --- Monte Carlo simulation ---
    N_SIMULATIONS = 10000
    team_qualify_count = {team: 0 for group in groups.values() for team in group}
    team_group_winner  = {team: 0 for group in groups.values() for team in group}
    team_points_total  = {team: 0 for group in groups.values() for team in group}

    for sim in range(N_SIMULATIONS):
        all_third_place = []

        for group_name, teams in groups.items():
            points = {team: 0 for team in teams}
            gd     = {team: 0 for team in teams}
            gf     = {team: 0 for team in teams}

            for i, team_a in enumerate(teams):
                for team_b in teams[i+1:]:

                    # If match already played, use real result
                    if (team_a, team_b) in completed:
                        hs, as_ = completed[(team_a, team_b)]
                        if hs > as_:
                            points[team_a] += 3
                        elif hs == as_:
                            points[team_a] += 1
                            points[team_b] += 1
                        else:
                            points[team_b] += 3
                        gd[team_a] += hs - as_
                        gd[team_b] += as_ - hs
                        gf[team_a] += hs
                        gf[team_b] += as_
                        continue

                    elif (team_b, team_a) in completed:
                        hs, as_ = completed[(team_b, team_a)]
                        if hs > as_:
                            points[team_b] += 3
                        elif hs == as_:
                            points[team_a] += 1
                            points[team_b] += 1
                        else:
                            points[team_a] += 3
                        gd[team_b] += hs - as_
                        gd[team_a] += as_ - hs
                        gf[team_b] += hs
                        gf[team_a] += as_
                        continue

                    # Otherwise simulate
                    if (team_a, team_b) in match_probs:
                        p_a_win, p_draw, p_b_win = match_probs[(team_a, team_b)]
                    elif (team_b, team_a) in match_probs:
                        p_b_win, p_draw, p_a_win = match_probs[(team_b, team_a)]
                    else:
                        p_a_win, p_draw, p_b_win = 0.4, 0.2, 0.4

                    total   = p_a_win + p_draw + p_b_win
                    p_a_win /= total
                    p_draw  /= total
                    p_b_win /= total

                    outcome = np.random.choice(
                        ['a_win', 'draw', 'b_win'],
                        p=[p_a_win, p_draw, p_b_win]
                    )

                    if outcome == 'a_win':
                        a_goals = np.random.choice([1,2,3], p=[0.45,0.35,0.20])
                        b_goals = np.random.choice([0,1],   p=[0.65,0.35])
                    elif outcome == 'draw':
                        g = np.random.choice([0,1,2], p=[0.30,0.45,0.25])
                        a_goals = b_goals = g
                    else:
                        b_goals = np.random.choice([1,2,3], p=[0.45,0.35,0.20])
                        a_goals = np.random.choice([0,1],   p=[0.65,0.35])

                    if outcome == 'a_win':
                        points[team_a] += 3
                    elif outcome == 'draw':
                        points[team_a] += 1
                        points[team_b] += 1
                    else:
                        points[team_b] += 3

                    gd[team_a] += a_goals - b_goals
                    gd[team_b] += b_goals - a_goals
                    gf[team_a] += a_goals
                    gf[team_b] += b_goals

            standings = sorted(teams, key=lambda t: (points[t], gd[t], gf[t]), reverse=True)
            team_qualify_count[standings[0]] += 1
            team_qualify_count[standings[1]] += 1
            team_group_winner[standings[0]]  += 1

            for t in teams:
                team_points_total[t] += points[t]

            third = standings[2]
            all_third_place.append((points[third], gd[third], gf[third], third))

        all_third_place.sort(reverse=True)
        for _, _, _, team in all_third_place[:8]:
            team_qualify_count[team] += 1

    results = []
    for group_name, teams in groups.items():
        for team in teams:
            results.append({
                'group':             group_name,
                'team':              team,
                'qualify_prob':      round(team_qualify_count[team] / N_SIMULATIONS * 100, 1),
                'group_winner_prob': round(team_group_winner[team]  / N_SIMULATIONS * 100, 1),
                'avg_points':        round(team_points_total[team]  / N_SIMULATIONS, 1),
            })

    return results, actual_results, list(groups.keys()), groups


HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WC 2026 Predictor</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #f9f9f7; color: #1a1a1a; }
  .container { max-width: 1100px; margin: 0 auto; padding: 2rem 1.5rem; }
  h1 { font-size: 22px; font-weight: 500; margin-bottom: 4px; }
  .subtitle { font-size: 13px; color: #888; margin-bottom: 1.5rem; }
  .tabs { display: flex; gap: 8px; margin-bottom: 1.5rem; flex-wrap: wrap; }
  .tab { padding: 6px 14px; font-size: 13px; border: 0.5px solid #ccc;
         border-radius: 8px; cursor: pointer; background: white; color: #555; }
  .tab.active { background: #1a1a1a; color: white; border-color: #1a1a1a; }
  .section { display: none; }
  .section.active { display: block; }
  .groups-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }
  .group-card { background: white; border: 0.5px solid #e0e0e0; border-radius: 12px; padding: 1rem 1.25rem; }
  .group-label { font-size: 11px; font-weight: 500; color: #888; letter-spacing: 0.08em;
                 text-transform: uppercase; margin-bottom: 10px; }
  .team-row { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
  .team-name { font-size: 13px; width: 130px; flex-shrink: 0; white-space: nowrap;
               overflow: hidden; text-overflow: ellipsis; }
  .bar-wrap { flex: 1; background: #f0f0ee; border-radius: 3px; height: 6px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 3px; }
  .bar-high { background: #1D9E75; }
  .bar-mid  { background: #378ADD; }
  .bar-low  { background: #aaa; }
  .pct { font-size: 12px; color: #888; width: 38px; text-align: right; flex-shrink: 0; }
  .badge { font-size: 10px; padding: 2px 6px; border-radius: 3px; flex-shrink: 0; }
  .badge-auto  { background: #E1F5EE; color: #0F6E56; }
  .badge-third { background: #E6F1FB; color: #185FA5; }
  .badge-out   { background: #f0f0f0; color: #aaa; }
  .overall-list { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 6px; }
  .overall-row { display: flex; align-items: center; gap: 10px; padding: 8px 12px;
                 background: white; border: 0.5px solid #e0e0e0; border-radius: 8px; }
  .rank-num { font-size: 12px; color: #aaa; width: 22px; flex-shrink: 0; }
  .overall-name { font-size: 13px; flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .overall-bar-wrap { width: 80px; background: #f0f0ee; border-radius: 3px; height: 5px;
                      overflow: hidden; flex-shrink: 0; }
  .overall-pct { font-size: 12px; color: #888; width: 40px; text-align: right; flex-shrink: 0; }
  .results-section { margin-bottom: 2rem; }
  .results-section h2 { font-size: 15px; font-weight: 500; margin-bottom: 10px; }
  .result-form { display: flex; gap: 8px; flex-wrap: wrap; align-items: center;
                 background: white; border: 0.5px solid #e0e0e0; border-radius: 12px;
                 padding: 1rem 1.25rem; margin-bottom: 1rem; }
  .result-form select, .result-form input[type=number] {
    padding: 6px 10px; border: 0.5px solid #ccc; border-radius: 8px;
    font-size: 13px; background: white; }
  .result-form input[type=number] { width: 60px; }
  .result-form button { padding: 6px 14px; background: #1a1a1a; color: white;
                        border: none; border-radius: 8px; font-size: 13px; cursor: pointer; }
  .result-form button:hover { background: #333; }
  .results-list { display: flex; flex-direction: column; gap: 6px; }
  .result-row { display: flex; align-items: center; gap: 10px; padding: 8px 12px;
                background: white; border: 0.5px solid #e0e0e0; border-radius: 8px;
                font-size: 13px; }
  .result-score { font-weight: 500; padding: 2px 10px; background: #f0f0ee; border-radius: 6px; }
  .result-teams { flex: 1; }
  .delete-btn { background: none; border: none; color: #ccc; cursor: pointer; font-size: 16px; }
  .delete-btn:hover { color: #e24b4a; }
  .accuracy-box { background: white; border: 0.5px solid #e0e0e0; border-radius: 12px;
                  padding: 1rem 1.25rem; margin-top: 1rem; }
  .acc-row { display: flex; justify-content: space-between; font-size: 13px; padding: 4px 0; }
  .acc-correct { color: #1D9E75; font-weight: 500; }
  .acc-wrong   { color: #e24b4a; font-weight: 500; }
  .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
                  gap: 12px; margin-bottom: 1.5rem; }
  .metric-card { background: #f0f0ee; border-radius: 8px; padding: 1rem; }
  .metric-label { font-size: 12px; color: #888; margin-bottom: 4px; }
  .metric-value { font-size: 24px; font-weight: 500; }
  .metric-sub { font-size: 12px; color: #888; margin-top: 2px; }
  .spinner { text-align: center; padding: 2rem; color: #888; font-size: 14px; }
</style>
</head>
<body>
<div class="container">
  <h1>World Cup 2026 — group stage predictor</h1>
  <p class="subtitle" id="subtitle">Loading simulation...</p>

  <div class="tabs">
    <button class="tab active" onclick="showTab('groups',this)">By group</button>
    <button class="tab" onclick="showTab('overall',this)">Overall ranking</button>
    <button class="tab" onclick="showTab('results',this)">Enter results</button>
    <button class="tab" onclick="showTab('model',this)">Model stats</button>
  </div>

  <div id="groups"  class="section active"><div class="spinner">Loading...</div></div>
  <div id="overall" class="section"><div class="spinner">Loading...</div></div>
  <div id="results" class="section">
    <div class="results-section">
      <h2>Add a result</h2>
      <div class="result-form">
        <select id="home-team"><option value="">Home team...</option></select>
        <input type="number" id="home-score" min="0" max="20" placeholder="0">
        <span style="font-size:13px;color:#888;">vs</span>
        <input type="number" id="away-score" min="0" max="20" placeholder="0">
        <select id="away-team"><option value="">Away team...</option></select>
        <button onclick="addResult()">Add result</button>
      </div>
      <div id="results-list" class="results-list"></div>
      <div id="accuracy-box" class="accuracy-box" style="display:none;">
        <div style="font-size:13px;font-weight:500;margin-bottom:8px;">Prediction accuracy</div>
        <div id="accuracy-rows"></div>
      </div>
    </div>
  </div>

  <div id="model" class="section">
    <div class="metrics-grid">
      <div class="metric-card"><div class="metric-label">Test accuracy</div>
        <div class="metric-value">60.5%</div><div class="metric-sub">vs 47.7% baseline</div></div>
      <div class="metric-card"><div class="metric-label">Log loss</div>
        <div class="metric-value">0.869</div><div class="metric-sub">lower is better</div></div>
      <div class="metric-card"><div class="metric-label">Training matches</div>
        <div class="metric-value">25,207</div><div class="metric-sub">1990–2026</div></div>
      <div class="metric-card"><div class="metric-label">Simulations</div>
        <div class="metric-value">10,000</div><div class="metric-sub">Monte Carlo runs</div></div>
    </div>
    <div style="background:white;border:0.5px solid #e0e0e0;border-radius:12px;padding:1rem 1.25rem;">
      <div style="font-size:13px;font-weight:500;margin-bottom:12px;">Feature importance</div>
      <div id="features"></div>
    </div>
  </div>
</div>

<script>
const features = [
  ['Elo difference',33.5],['Avg goals conceded diff',10.4],['Is competitive',6.5],
  ['Away Elo',6.1],['Is World Cup',6.0],['Home Elo',5.8],['Home avg conceded',5.1],
  ['Away avg conceded',4.7],['Avg scored diff',4.5],['Form diff',4.0],
  ['Away form',3.6],['Home avg scored',3.3],['Home form',3.2],['Away avg scored',3.2],
  ['Home advantage',0.1]
];

let simData = [];
let allTeams = [];
let matchProbs = {};
let actualResults = [];

function barClass(p) {
  return p >= 75 ? 'bar-high' : p >= 50 ? 'bar-mid' : 'bar-low';
}

function badge(i, p) {
  if (i === 0) return '<span class="badge badge-auto">1st</span>';
  if (i === 1) return '<span class="badge badge-auto">2nd</span>';
  if (p >= 45) return '<span class="badge badge-third">possible</span>';
  return '<span class="badge badge-out">out</span>';
}

function renderGroups(data) {
  const byGroup = {};
  data.forEach(t => { (byGroup[t.group] = byGroup[t.group] || []).push(t); });
  const grid = document.createElement('div');
  grid.className = 'groups-grid';
  Object.entries(byGroup).sort().forEach(([g, teams]) => {
    teams.sort((a,b) => b.qualify_prob - a.qualify_prob);
    let html = `<div class="group-label">Group ${g}</div>`;
    teams.forEach((t, i) => {
      html += `<div class="team-row">
        <span class="team-name">${t.team}</span>
        <div class="bar-wrap"><div class="bar-fill ${barClass(t.qualify_prob)}" style="width:${t.qualify_prob}%"></div></div>
        <span class="pct">${t.qualify_prob.toFixed(1)}%</span>
        ${badge(i, t.qualify_prob)}
      </div>`;
    });
    const card = document.createElement('div');
    card.className = 'group-card';
    card.innerHTML = html;
    grid.appendChild(card);
  });
  const el = document.getElementById('groups');
  el.innerHTML = '';
  el.appendChild(grid);
}

function renderOverall(data) {
  const sorted = [...data].sort((a,b) => b.qualify_prob - a.qualify_prob);
  const el = document.getElementById('overall');
  el.innerHTML = '<div class="overall-list">' + sorted.map((t,i) =>
    `<div class="overall-row">
      <span class="rank-num">${i+1}</span>
      <span class="overall-name">${t.team}</span>
      <span style="font-size:11px;color:#aaa;flex-shrink:0;">Grp ${t.group}</span>
      <div class="overall-bar-wrap"><div class="bar-fill ${barClass(t.qualify_prob)}" style="width:${t.qualify_prob}%"></div></div>
      <span class="overall-pct">${t.qualify_prob.toFixed(1)}%</span>
    </div>`
  ).join('') + '</div>';
}

function renderFeatures() {
  const max = features[0][1];
  document.getElementById('features').innerHTML = features.map(([name, imp]) =>
    `<div class="team-row" style="margin-bottom:8px;">
      <span class="team-name" style="font-size:12px;width:180px;">${name}</span>
      <div class="bar-wrap"><div class="bar-fill bar-mid" style="width:${(imp/max*100).toFixed(1)}%"></div></div>
      <span class="pct" style="font-size:12px;">${imp.toFixed(1)}%</span>
    </div>`
  ).join('');
}

function populateTeamDropdowns(teams) {
  ['home-team','away-team'].forEach(id => {
    const sel = document.getElementById(id);
    sel.innerHTML = `<option value="">Team...</option>` +
      teams.map(t => `<option value="${t}">${t}</option>`).join('');
  });
}

function renderResults() {
  const list = document.getElementById('results-list');
  list.innerHTML = actualResults.length === 0
    ? '<p style="font-size:13px;color:#aaa;padding:8px 0;">No results added yet.</p>'
    : actualResults.map((r, i) =>
        `<div class="result-row">
          <span class="result-teams">${r.home_team} vs ${r.away_team}</span>
          <span class="result-score">${r.home_score} – ${r.away_score}</span>
          <button class="delete-btn" onclick="deleteResult(${i})">×</button>
        </div>`
      ).join('');
  renderAccuracy();
}

function renderAccuracy() {
  if (actualResults.length === 0) {
    document.getElementById('accuracy-box').style.display = 'none';
    return;
  }
  document.getElementById('accuracy-box').style.display = 'block';
  let correct = 0, total = 0;
  let rows = '';
  actualResults.forEach(r => {
    const key1 = r.home_team + '|' + r.away_team;
    const key2 = r.away_team + '|' + r.home_team;
    let prob = matchProbs[key1] || (matchProbs[key2] ? {
      p_home_win: matchProbs[key2].p_away_win,
      p_draw: matchProbs[key2].p_draw,
      p_away_win: matchProbs[key2].p_home_win
    } : null);
    if (!prob) return;
    const probs = [prob.p_away_win, prob.p_draw, prob.p_home_win];
    const predicted = probs.indexOf(Math.max(...probs));
    const hs = r.home_score, as_ = r.away_score;
    const actual = hs > as_ ? 2 : hs === as_ ? 1 : 0;
    const labels = ['Away win','Draw','Home win'];
    const isCorrect = predicted === actual;
    if (isCorrect) correct++;
    total++;
    rows += `<div class="acc-row">
      <span>${r.home_team} vs ${r.away_team}</span>
      <span>Predicted: ${labels[predicted]}</span>
      <span>Actual: ${labels[actual]}</span>
      <span class="${isCorrect ? 'acc-correct' : 'acc-wrong'}">${isCorrect ? '✓' : '✗'}</span>
    </div>`;
  });
  document.getElementById('accuracy-rows').innerHTML =
    `<div class="acc-row" style="font-weight:500;margin-bottom:8px;">
      <span>${correct}/${total} correct</span>
      <span>${total > 0 ? (correct/total*100).toFixed(1) : 0}% accuracy</span>
    </div>` + rows;
}

async function addResult() {
  const ht = document.getElementById('home-team').value;
  const at = document.getElementById('away-team').value;
  const hs = parseInt(document.getElementById('home-score').value);
  const as_ = parseInt(document.getElementById('away-score').value);
  if (!ht || !at || ht === at || isNaN(hs) || isNaN(as_)) {
    alert('Please fill in all fields correctly.'); return;
  }
  const res = await fetch('/add_result', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({home_team:ht, away_team:at, home_score:hs, away_score:as_})
  });
  const data = await res.json();
  if (data.status === 'ok') {
    actualResults = data.actual_results;
    simData = data.sim_results;
    renderGroups(simData);
    renderOverall(simData);
    renderResults();
    document.getElementById('subtitle').textContent =
      `${actualResults.length} result(s) entered · simulation updated · 10,000 runs`;
  }
}

async function deleteResult(i) {
  const res = await fetch('/delete_result', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({index: i})
  });
  const data = await res.json();
  if (data.status === 'ok') {
    actualResults = data.actual_results;
    simData = data.sim_results;
    renderGroups(simData);
    renderOverall(simData);
    renderResults();
  }
}

function showTab(id, btn) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  btn.classList.add('active');
}

async function init() {
  const res = await fetch('/data');
  const data = await res.json();
  simData = data.sim_results;
  allTeams = data.all_teams;
  matchProbs = data.match_probs;
  actualResults = data.actual_results;
  renderGroups(simData);
  renderOverall(simData);
  renderFeatures();
  populateTeamDropdowns(allTeams);
  renderResults();
  document.getElementById('subtitle').textContent =
    `${actualResults.length} result(s) entered · Monte Carlo simulation · 10,000 runs · XGBoost · 60.5% accuracy`;
}

init();
</script>
</body>
</html>'''


@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/data')
def data():
    sim_results, actual_results, group_keys, groups = run_simulation()

    model        = joblib.load(os.path.join(DATA_DIR, 'model.pkl'))
    feature_cols = joblib.load(os.path.join(DATA_DIR, 'feature_cols.pkl'))
    features     = pd.read_csv(os.path.join(DATA_DIR, 'features.csv'), parse_dates=['date'])

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
        key = f"{row['home_team']}|{row['away_team']}"
        match_probs[key] = {
            'p_home_win': round(row['p_home_win'], 4),
            'p_draw':     round(row['p_draw'], 4),
            'p_away_win': round(row['p_away_win'], 4),
        }

    all_teams = sorted(set(t for g in groups.values() for t in g))

    return jsonify({
        'sim_results':    sim_results,
        'actual_results': actual_results,
        'match_probs':    match_probs,
        'all_teams':      all_teams,
    })


@app.route('/add_result', methods=['POST'])
def add_result():
    from flask import request
    body = request.get_json()

    path = os.path.join(DATA_DIR, 'actual_results.json')
    results = []
    if os.path.exists(path):
        with open(path) as f:
            results = json.load(f)

    results.append({
        'home_team':  body['home_team'],
        'away_team':  body['away_team'],
        'home_score': body['home_score'],
        'away_score': body['away_score'],
    })

    with open(path, 'w') as f:
        json.dump(results, f)

    sim_results, actual_results, _, _ = run_simulation()
    return jsonify({'status': 'ok', 'sim_results': sim_results, 'actual_results': actual_results})


@app.route('/delete_result', methods=['POST'])
def delete_result():
    from flask import request
    body = request.get_json()

    path = os.path.join(DATA_DIR, 'actual_results.json')
    results = []
    if os.path.exists(path):
        with open(path) as f:
            results = json.load(f)

    idx = body['index']
    if 0 <= idx < len(results):
        results.pop(idx)

    with open(path, 'w') as f:
        json.dump(results, f)

    sim_results, actual_results, _, _ = run_simulation()
    return jsonify({'status': 'ok', 'sim_results': sim_results, 'actual_results': actual_results})


if __name__ == '__main__':
    app.run(debug=True, port=5000)