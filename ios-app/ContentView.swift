//
//  ContentView.swift
//  WorldCupPredictor
//

import SwiftUI
import Combine

// MARK: - Data Models

struct MatchProb: Codable {
    let home_team: String
    let away_team: String
    let p_home_win: Double
    let p_draw: Double
    let p_away_win: Double
}

struct AppData: Codable {
    let matches: [MatchProb]
    let groups: [String: [String]]
    let elo_ratings: [String: Double]
}

struct ActualResult: Codable, Identifiable {
    let id: UUID
    let homeTeam: String
    let awayTeam: String
    let homeScore: Int
    let awayScore: Int

    init(homeTeam: String, awayTeam: String, homeScore: Int, awayScore: Int) {
        self.id = UUID()
        self.homeTeam = homeTeam
        self.awayTeam = awayTeam
        self.homeScore = homeScore
        self.awayScore = awayScore
    }
}

struct TeamResult: Identifiable {
    let id = UUID()
    let team: String
    let group: String
    let qualifyProb: Double
    let winGroupProb: Double
    let avgPoints: Double
}

// MARK: - Live Scores Manager

class LiveScoresManager: ObservableObject {
    @Published var lastUpdated: Date? = nil
    @Published var isUpdating = false

    let apiKey = Secrets.footballDataAPIKey
    let competitionId = "2000"

    func fetchLatestResults(completion: @escaping ([ActualResult]) -> Void) {
        isUpdating = true

        let urlString = "https://api.football-data.org/v4/competitions/\(competitionId)/matches?status=FINISHED"

        guard let url = URL(string: urlString) else { return }

        var request = URLRequest(url: url)
        request.setValue(apiKey, forHTTPHeaderField: "X-Auth-Token")

        URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            guard let self = self else { return }

            defer {
                DispatchQueue.main.async { self.isUpdating = false }
            }

            guard let data = data, error == nil else {
                print("❌ API error: \(error?.localizedDescription ?? "unknown")")
                return
            }

            guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let matches = json["matches"] as? [[String: Any]] else {
                print("❌ Could not parse API response")
                return
            }

            var results: [ActualResult] = []

            for match in matches {
                guard
                    let homeTeamDict = match["homeTeam"] as? [String: Any],
                    let awayTeamDict = match["awayTeam"] as? [String: Any],
                    let scoreDict    = match["score"] as? [String: Any],
                    let fullTime     = scoreDict["fullTime"] as? [String: Any],
                    let homeScore    = fullTime["home"] as? Int,
                    let awayScore    = fullTime["away"] as? Int,
                    let homeTeamRaw  = homeTeamDict["name"] as? String,
                    let awayTeamRaw  = awayTeamDict["name"] as? String
                else { continue }

                let homeTeam = normaliseTeamName(homeTeamRaw)
                let awayTeam = normaliseTeamName(awayTeamRaw)

                results.append(ActualResult(
                    homeTeam: homeTeam,
                    awayTeam: awayTeam,
                    homeScore: homeScore,
                    awayScore: awayScore
                ))
            }

            DispatchQueue.main.async {
                self.lastUpdated = Date()
                completion(results)
            }
        }.resume()
    }
}

func normaliseTeamName(_ name: String) -> String {
    let mapping: [String: String] = [
        "United States": "United States",
        "USA": "United States",
        "Korea Republic": "South Korea",
        "IR Iran": "Iran",
        "Côte d'Ivoire": "Ivory Coast",
        "Bosnia and Herzegovina": "Bosnia and Herzegovina",
        "Türkiye": "Turkey",
        "DR Congo": "DR Congo",
        "Curaçao": "Curaçao",
        "Cape Verde": "Cape Verde",
        "Saudi Arabia": "Saudi Arabia",
        "New Zealand": "New Zealand",
    ]
    return mapping[name] ?? name
}

func timeAgo(_ date: Date) -> String {
    let seconds = Int(Date().timeIntervalSince(date))
    if seconds < 60 { return "just now" }
    if seconds < 3600 { return "\(seconds / 60)m ago" }
    return "\(seconds / 3600)h ago"
}

// MARK: - Prediction helper

func smartPredict(pHome: Double, pDraw: Double, pAway: Double) -> Int {
    let gap = abs(pHome - pAway)
    if pDraw > 0.30 && gap < 0.18 {
        return 1
    } else if pHome > pAway {
        return 2
    } else {
        return 0
    }
}

// MARK: - Simulator

class Simulator: ObservableObject {
    @Published var teamResults: [TeamResult] = []
    @Published var actualResults: [ActualResult] = []
    @Published var isSimulating = false

    var appData: AppData?
    var matchProbs: [String: MatchProb] = [:]
    @Published var knockoutData: KnockoutData?
    var liveScores = LiveScoresManager()
    var refreshTimer: Timer?

    init() {
        loadData()
        loadKnockoutData()
        loadSavedResults()
        startAutoRefresh()
    }
    
    func loadKnockoutData() {
        guard let url = Bundle.main.url(forResource: "knockout_results", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let decoded = try? JSONDecoder().decode(KnockoutData.self, from: data) else {
            print("❌ Could not load knockout_results.json")
            return
        }
        knockoutData = decoded
        print("✅ Loaded knockout data: \(decoded.round_of_16_probs.count) matches")
    }

    func startAutoRefresh() {
        fetchAndUpdate()
        refreshTimer = Timer.scheduledTimer(withTimeInterval: 180, repeats: true) { [weak self] _ in
            self?.fetchAndUpdate()
        }
    }

    func stopAutoRefresh() {
        refreshTimer?.invalidate()
        refreshTimer = nil
    }

    func fetchAndUpdate() {
        liveScores.fetchLatestResults { [weak self] results in
            guard let self = self else { return }
            if !results.isEmpty {
                self.actualResults = results
                self.saveResults()
                self.runSimulation()
                print("✅ Auto-updated with \(results.count) results")
            } else {
                // Fall back to saved results and run simulation
                self.runSimulation()
            }
        }
    }

    func loadData() {
        guard let url = Bundle.main.url(forResource: "app_data", withExtension: "json") else {
            print("❌ app_data.json not found in bundle")
            appData = AppData(matches: [], groups: [:], elo_ratings: [:])
            return
        }
        guard let data = try? Data(contentsOf: url) else {
            print("❌ Could not read app_data.json")
            appData = AppData(matches: [], groups: [:], elo_ratings: [:])
            return
        }
        guard let decoded = try? JSONDecoder().decode(AppData.self, from: data) else {
            print("❌ Could not decode app_data.json")
            appData = AppData(matches: [], groups: [:], elo_ratings: [:])
            return
        }
        appData = decoded
        for m in decoded.matches {
            let key = "\(m.home_team)|\(m.away_team)"
            matchProbs[key] = m
        }
        print("✅ Loaded \(decoded.matches.count) matches")
    }

    func getProbs(teamA: String, teamB: String) -> (Double, Double, Double) {
        let key1 = "\(teamA)|\(teamB)"
        if let m = matchProbs[key1] {
            return (m.p_home_win, m.p_draw, m.p_away_win)
        }
        let key2 = "\(teamB)|\(teamA)"
        if let m = matchProbs[key2] {
            return (m.p_away_win, m.p_draw, m.p_home_win)
        }
        return (0.4, 0.2, 0.4)
    }

    func runSimulation() {
        guard let data = appData else { return }
        isSimulating = true

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }

            let completed = self.buildCompleted()
            let N = 10000
            var qualifyCount: [String: Int] = [:]
            var winnerCount: [String: Int] = [:]
            var pointsTotal: [String: Double] = [:]

            for teams in data.groups.values {
                for t in teams {
                    qualifyCount[t] = 0
                    winnerCount[t] = 0
                    pointsTotal[t] = 0
                }
            }

            for _ in 0..<N {
                var allThirds: [(Int, Int, Int, String)] = []

                for (_, teams) in data.groups {
                    var pts: [String: Int] = [:]
                    var gd: [String: Int] = [:]
                    var gf: [String: Int] = [:]
                    for t in teams { pts[t] = 0; gd[t] = 0; gf[t] = 0 }

                    for i in 0..<teams.count {
                        for j in (i+1)..<teams.count {
                            let ta = teams[i], tb = teams[j]

                            let keyAB = "\(ta)|\(tb)"
                            let keyBA = "\(tb)|\(ta)"
                            var hScore: Int, aScore: Int
                            var usedReal = false

                            if completed[keyAB] != nil {
                                hScore = completed[keyAB]!.1
                                aScore = completed[keyAB]!.2
                                usedReal = true
                            } else if completed[keyBA] != nil {
                                hScore = completed[keyBA]!.2
                                aScore = completed[keyBA]!.1
                                usedReal = true
                            } else {
                                hScore = 0; aScore = 0
                            }

                            if usedReal {
                                if hScore > aScore { pts[ta]! += 3 }
                                else if hScore == aScore { pts[ta]! += 1; pts[tb]! += 1 }
                                else { pts[tb]! += 3 }
                                gd[ta]! += hScore - aScore
                                gd[tb]! += aScore - hScore
                                gf[ta]! += hScore
                                gf[tb]! += aScore
                                continue
                            }

                            let (pA, pD, pB) = self.getProbs(teamA: ta, teamB: tb)
                            let total = pA + pD + pB
                            let r = Double.random(in: 0..<1)
                            let normA = pA / total
                            let normD = pD / total

                            var aGoals: Int, bGoals: Int
                            if r < normA {
                                aGoals = [1,2,3].randomElement(weights: [0.45,0.35,0.20])
                                bGoals = [0,1].randomElement(weights: [0.65,0.35])
                                pts[ta]! += 3
                            } else if r < normA + normD {
                                let g = [0,1,2].randomElement(weights: [0.30,0.45,0.25])
                                aGoals = g; bGoals = g
                                pts[ta]! += 1; pts[tb]! += 1
                            } else {
                                bGoals = [1,2,3].randomElement(weights: [0.45,0.35,0.20])
                                aGoals = [0,1].randomElement(weights: [0.65,0.35])
                                pts[tb]! += 3
                            }

                            gd[ta]! += aGoals - bGoals
                            gd[tb]! += bGoals - aGoals
                            gf[ta]! += aGoals
                            gf[tb]! += bGoals
                        }
                    }

                    let standings = teams.sorted {
                        (pts[$0]!, gd[$0]!, gf[$0]!) > (pts[$1]!, gd[$1]!, gf[$1]!)
                    }

                    qualifyCount[standings[0]]! += 1
                    qualifyCount[standings[1]]! += 1
                    winnerCount[standings[0]]! += 1

                    for t in teams { pointsTotal[t]! += Double(pts[t]!) }

                    let third = standings[2]
                    allThirds.append((pts[third]!, gd[third]!, gf[third]!, third))
                }

                allThirds.sort { ($0.0, $0.1, $0.2) > ($1.0, $1.1, $1.2) }
                for k in 0..<min(8, allThirds.count) {
                    qualifyCount[allThirds[k].3]! += 1
                }
            }

            var results: [TeamResult] = []
            for (g, teams) in data.groups {
                for t in teams {
                    results.append(TeamResult(
                        team: t, group: g,
                        qualifyProb: Double(qualifyCount[t]!) / Double(N) * 100,
                        winGroupProb: Double(winnerCount[t]!) / Double(N) * 100,
                        avgPoints: pointsTotal[t]! / Double(N)
                    ))
                }
            }

            DispatchQueue.main.async {
                self.teamResults = results
                self.isSimulating = false
            }
        }
    }

    func buildCompleted() -> [String: (String, Int, Int)] {
        var dict: [String: (String, Int, Int)] = [:]
        for r in actualResults {
            dict["\(r.homeTeam)|\(r.awayTeam)"] = (r.homeTeam, r.homeScore, r.awayScore)
        }
        return dict
    }

    func addResult(home: String, away: String, homeScore: Int, awayScore: Int) {
        actualResults.append(ActualResult(
            homeTeam: home, awayTeam: away,
            homeScore: homeScore, awayScore: awayScore))
        saveResults()
        runSimulation()
    }

    func deleteResult(at index: Int) {
        actualResults.remove(at: index)
        saveResults()
        runSimulation()
    }

    func saveResults() {
        if let data = try? JSONEncoder().encode(actualResults) {
            UserDefaults.standard.set(data, forKey: "actual_results")
        }
    }

    func loadSavedResults() {
        if let data = UserDefaults.standard.data(forKey: "actual_results"),
           let decoded = try? JSONDecoder().decode([ActualResult].self, from: data) {
            actualResults = decoded
        }
    }
}

// MARK: - Helpers

extension Array where Element == Int {
    func randomElement(weights: [Double]) -> Int {
        let r = Double.random(in: 0..<1)
        var cumulative = 0.0
        for (i, w) in weights.enumerated() {
            cumulative += w
            if r < cumulative { return self[i] }
        }
        return self.last!
    }
}

// MARK: - Content View

struct ContentView: View {
    @StateObject var sim = Simulator()

    var body: some View {
        TabView {
            GroupsView(sim: sim)
                .tabItem { Label("Groups", systemImage: "rectangle.3.group") }
            RankingView(sim: sim)
                .tabItem { Label("Ranking", systemImage: "list.number") }
            MatchesView(sim: sim)
                .tabItem { Label("Matches", systemImage: "sportscourt") }
            ResultsView(sim: sim)
                .tabItem { Label("Results", systemImage: "plus.circle") }
            AccuracyView(sim: sim)
                .tabItem { Label("Accuracy", systemImage: "checkmark.circle") }
            KnockoutView(sim: sim)
                .tabItem { Label("Knockout", systemImage: "trophy") }
        }
    }
}

// MARK: - Groups View

struct GroupsView: View {
    @ObservedObject var sim: Simulator

    var body: some View {
        NavigationView {
            ScrollView {
                LazyVStack(spacing: 16) {
                    if sim.isSimulating {
                        ProgressView("Simulating 10,000 runs...")
                            .padding()
                    }

                    // Live update status banner
                    if sim.liveScores.isUpdating {
                        HStack {
                            ProgressView()
                                .scaleEffect(0.8)
                            Text("Fetching latest scores...")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        .padding(.vertical, 4)
                    } else if let updated = sim.liveScores.lastUpdated {
                        HStack {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundColor(.green)
                                .font(.caption)
                            Text("Scores updated \(timeAgo(updated))")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Spacer()
                            Text("\(sim.actualResults.count) results loaded")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        .padding(.horizontal, 4)
                        .padding(.vertical, 4)
                    }

                    ForEach(sortedGroups, id: \.0) { group, teams in
                        GroupCard(groupName: group, teams: teams)
                    }
                }
                .padding()
            }
            .navigationTitle("World Cup 2026")
            .background(Color(.systemGroupedBackground))
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: {
                        sim.fetchAndUpdate()
                    }) {
                        Image(systemName: "arrow.clockwise")
                    }
                }
            }
        }
    }

    var sortedGroups: [(String, [TeamResult])] {
        let grouped = Dictionary(grouping: sim.teamResults) { $0.group }
        return grouped.sorted { $0.key < $1.key }
            .map { ($0.key, $0.value.sorted { $0.qualifyProb > $1.qualifyProb }) }
    }
}

struct GroupCard: View {
    let groupName: String
    let teams: [TeamResult]

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("GROUP \(groupName)")
                .font(.caption)
                .fontWeight(.semibold)
                .foregroundColor(.secondary)
                .tracking(1)
            ForEach(Array(teams.enumerated()), id: \.1.id) { index, team in
                TeamRow(team: team, rank: index + 1)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
    }
}

struct TeamRow: View {
    let team: TeamResult
    let rank: Int

    var barColor: Color {
        if team.qualifyProb >= 75 { return .green }
        if team.qualifyProb >= 50 { return .blue }
        return .gray
    }

    var badge: String {
        if rank == 1 { return "1st" }
        if rank == 2 { return "2nd" }
        if team.qualifyProb >= 45 { return "3rd?" }
        return "out"
    }

    var badgeColor: Color {
        if rank <= 2 { return .green }
        if team.qualifyProb >= 45 { return .blue }
        return .gray
    }

    var body: some View {
        HStack(spacing: 8) {
            Text(team.team)
                .font(.subheadline)
                .frame(width: 120, alignment: .leading)
                .lineLimit(1)

            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 3)
                        .fill(Color(.systemGray5))
                    RoundedRectangle(cornerRadius: 3)
                        .fill(barColor)
                        .frame(width: geo.size.width * team.qualifyProb / 100)
                }
            }
            .frame(height: 6)

            Text(String(format: "%.1f%%", team.qualifyProb))
                .font(.caption)
                .foregroundColor(.secondary)
                .frame(width: 44, alignment: .trailing)

            Text(badge)
                .font(.system(size: 9, weight: .medium))
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(badgeColor.opacity(0.15))
                .foregroundColor(badgeColor)
                .cornerRadius(4)
        }
    }
}

// MARK: - Ranking View

struct RankingView: View {
    @ObservedObject var sim: Simulator

    var sorted: [TeamResult] {
        sim.teamResults.sorted { $0.qualifyProb > $1.qualifyProb }
    }

    var body: some View {
        NavigationView {
            List(Array(sorted.enumerated()), id: \.1.id) { index, team in
                HStack {
                    Text("\(index + 1)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .frame(width: 26, alignment: .trailing)

                    VStack(alignment: .leading, spacing: 2) {
                        Text(team.team)
                            .font(.subheadline)
                        Text("Group \(team.group) · \(String(format: "%.1f", team.avgPoints)) avg pts")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }

                    Spacer()

                    Text(String(format: "%.1f%%", team.qualifyProb))
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .foregroundColor(
                            team.qualifyProb >= 75 ? .green :
                            team.qualifyProb >= 50 ? .blue : .gray
                        )
                }
                .padding(.vertical, 2)
            }
            .navigationTitle("Overall ranking")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: {
                        sim.fetchAndUpdate()
                    }) {
                        Image(systemName: "arrow.clockwise")
                    }
                }
            }
        }
    }
}

// MARK: - Matches View

struct MatchesView: View {
    @ObservedObject var sim: Simulator

    let groupOrder = ["A","B","C","D","E","F","G","H","I","J","K","L"]

    var body: some View {
        NavigationView {
            List {
                ForEach(groupOrder, id: \.self) { group in
                    if let teams = sim.appData?.groups[group] {
                        Section("Group \(group)") {
                            ForEach(matchesForGroup(teams: teams), id: \.0) { key, home, away in
                                MatchProbRow(homeTeam: home, awayTeam: away, sim: sim)
                            }
                        }
                    }
                }
            }
            .navigationTitle("Match predictions")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: {
                        sim.fetchAndUpdate()
                    }) {
                        Image(systemName: "arrow.clockwise")
                    }
                }
            }
        }
    }

    func matchesForGroup(teams: [String]) -> [(String, String, String)] {
        var matches: [(String, String, String)] = []
        for i in 0..<teams.count {
            for j in (i+1)..<teams.count {
                matches.append(("\(teams[i])|\(teams[j])", teams[i], teams[j]))
            }
        }
        return matches
    }
}

struct MatchProbRow: View {
    let homeTeam: String
    let awayTeam: String
    @ObservedObject var sim: Simulator

    var probs: (Double, Double, Double) {
        sim.getProbs(teamA: homeTeam, teamB: awayTeam)
    }

    var result: ActualResult? {
        sim.actualResults.first {
            ($0.homeTeam == homeTeam && $0.awayTeam == awayTeam) ||
            ($0.homeTeam == awayTeam && $0.awayTeam == homeTeam)
        }
    }

    var predictedOutcome: Int {
        let (pH, pD, pA) = probs
        return smartPredict(pHome: pH, pDraw: pD, pAway: pA)
    }

    var predictedLabel: String {
        switch predictedOutcome {
        case 2: return homeTeam
        case 1: return "Draw"
        default: return awayTeam
        }
    }

    var isCorrect: Bool? {
        guard let r = result else { return nil }
        let actual: Int = r.homeScore > r.awayScore ? 2 :
                         r.homeScore == r.awayScore ? 1 : 0
        return predictedOutcome == actual
    }

    var body: some View {
        VStack(spacing: 8) {
            HStack {
                Text(homeTeam)
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .lineLimit(1)

                if let r = result {
                    Text("\(r.homeScore) – \(r.awayScore)")
                        .font(.subheadline)
                        .fontWeight(.bold)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 4)
                        .background(Color(.systemGray5))
                        .cornerRadius(8)
                } else {
                    Text("vs")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .padding(.horizontal, 8)
                }

                Text(awayTeam)
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .frame(maxWidth: .infinity, alignment: .trailing)
                    .lineLimit(1)
            }

            GeometryReader { geo in
                HStack(spacing: 2) {
                    Rectangle()
                        .fill(Color.green)
                        .frame(width: max(2, geo.size.width * probs.0))
                        .cornerRadius(3)
                    Rectangle()
                        .fill(Color.gray.opacity(0.35))
                        .frame(width: max(2, geo.size.width * probs.1))
                    Rectangle()
                        .fill(Color.blue)
                        .frame(width: max(2, geo.size.width * probs.2))
                        .cornerRadius(3)
                }
            }
            .frame(height: 8)
            .cornerRadius(4)

            HStack {
                Text(String(format: "%.0f%%", probs.0 * 100))
                    .font(.caption2)
                    .foregroundColor(.green)
                    .fontWeight(.medium)

                Spacer()

                Text(String(format: "Draw %.0f%%", probs.1 * 100))
                    .font(.caption2)
                    .foregroundColor(.secondary)

                Spacer()

                Text(String(format: "%.0f%%", probs.2 * 100))
                    .font(.caption2)
                    .foregroundColor(.blue)
                    .fontWeight(.medium)
            }

            HStack {
                Image(systemName: "brain")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                Text("Predicts: \(predictedLabel)")
                    .font(.caption2)
                    .foregroundColor(.secondary)

                Spacer()

                if let correct = isCorrect {
                    Text(correct ? "✓ Correct" : "✗ Wrong")
                        .font(.caption2)
                        .fontWeight(.medium)
                        .foregroundColor(correct ? .green : .red)
                }
            }
        }
        .padding(.vertical, 6)
    }
}

// MARK: - Results View

struct ResultsView: View {
    @ObservedObject var sim: Simulator
    @State private var homeTeam = ""
    @State private var awayTeam = ""
    @State private var homeScore = 0
    @State private var awayScore = 0

    var allTeams: [String] {
        sim.appData?.groups.values.flatMap { $0 }.sorted() ?? []
    }

    var body: some View {
        NavigationView {
            List {
                Section("Live scores") {
                    HStack {
                        VStack(alignment: .leading, spacing: 4) {
                            if sim.liveScores.isUpdating {
                                HStack {
                                    ProgressView().scaleEffect(0.7)
                                    Text("Fetching scores...")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                            } else if let updated = sim.liveScores.lastUpdated {
                                Text("Auto-updating every 3 minutes")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                Text("Last fetch: \(timeAgo(updated))")
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                            } else {
                                Text("Waiting for first fetch...")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                        Spacer()
                        Button("Refresh now") {
                            sim.fetchAndUpdate()
                        }
                        .font(.caption)
                        .buttonStyle(.bordered)
                    }
                    .padding(.vertical, 4)
                }

                Section("Add a result manually") {
                    Picker("Home team", selection: $homeTeam) {
                        Text("Select...").tag("")
                        ForEach(allTeams, id: \.self) { Text($0).tag($0) }
                    }

                    HStack {
                        Text("Score")
                        Spacer()
                        Stepper("\(homeScore)", value: $homeScore, in: 0...20)
                            .frame(width: 120)
                        Text("–")
                        Stepper("\(awayScore)", value: $awayScore, in: 0...20)
                            .frame(width: 120)
                    }

                    Picker("Away team", selection: $awayTeam) {
                        Text("Select...").tag("")
                        ForEach(allTeams, id: \.self) { Text($0).tag($0) }
                    }

                    Button("Add result") {
                        guard !homeTeam.isEmpty, !awayTeam.isEmpty,
                              homeTeam != awayTeam else { return }
                        sim.addResult(home: homeTeam, away: awayTeam,
                                      homeScore: homeScore, awayScore: awayScore)
                        homeTeam = ""; awayTeam = ""
                        homeScore = 0; awayScore = 0
                    }
                    .disabled(homeTeam.isEmpty || awayTeam.isEmpty || homeTeam == awayTeam)
                }

                Section("Results loaded (\(sim.actualResults.count))") {
                    if sim.actualResults.isEmpty {
                        Text("No results yet.")
                            .foregroundColor(.secondary)
                    }
                    ForEach(Array(sim.actualResults.enumerated()), id: \.1.id) { i, r in
                        HStack {
                            Text(r.homeTeam)
                                .frame(maxWidth: .infinity, alignment: .trailing)
                            Text("\(r.homeScore) – \(r.awayScore)")
                                .fontWeight(.semibold)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 3)
                                .background(Color(.systemGray5))
                                .cornerRadius(6)
                            Text(r.awayTeam)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .font(.subheadline)
                        .swipeActions(edge: .trailing) {
                            Button(role: .destructive) {
                                sim.deleteResult(at: i)
                            } label: {
                                Label("Delete", systemImage: "trash")
                            }
                        }
                    }
                }
            }
            .navigationTitle("Results")
        }
    }
}

// MARK: - Accuracy View

struct AccuracyView: View {
    @ObservedObject var sim: Simulator

    var predictions: [(ActualResult, String, String, Bool)] {
        var results: [(ActualResult, String, String, Bool)] = []
        let labels = [0: "Away win", 1: "Draw", 2: "Home win"]

        for r in sim.actualResults {
            let (pH, pD, pA) = sim.getProbs(teamA: r.homeTeam, teamB: r.awayTeam)
            let predicted = smartPredict(pHome: pH, pDraw: pD, pAway: pA)

            let actual: Int
            if r.homeScore > r.awayScore { actual = 2 }
            else if r.homeScore == r.awayScore { actual = 1 }
            else { actual = 0 }

            results.append((r, labels[predicted]!, labels[actual]!, predicted == actual))
        }
        return results
    }

    var correctCount: Int { predictions.filter { $0.3 }.count }
    var total: Int { predictions.count }
    var accuracy: Double { total > 0 ? Double(correctCount) / Double(total) * 100 : 0 }

    var body: some View {
        NavigationView {
            List {
                if total > 0 {
                    Section {
                        HStack {
                            VStack(alignment: .leading) {
                                Text(String(format: "%.1f%%", accuracy))
                                    .font(.title)
                                    .fontWeight(.medium)
                                Text("\(correctCount)/\(total) correct")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            Spacer()
                            Image(systemName: accuracy >= 50 ?
                                  "checkmark.circle.fill" : "xmark.circle.fill")
                                .font(.title)
                                .foregroundColor(accuracy >= 50 ? .green : .red)
                        }
                        .padding(.vertical, 4)
                    }

                    Section("Match by match") {
                        ForEach(predictions, id: \.0.id) { r, pred, actual, correct in
                            VStack(alignment: .leading, spacing: 4) {
                                Text("\(r.homeTeam) \(r.homeScore)–\(r.awayScore) \(r.awayTeam)")
                                    .font(.subheadline)
                                HStack {
                                    Text("Predicted: \(pred)")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                    Spacer()
                                    Text(correct ? "✓ Correct" : "✗ Wrong")
                                        .font(.caption)
                                        .fontWeight(.medium)
                                        .foregroundColor(correct ? .green : .red)
                                }
                            }
                            .padding(.vertical, 2)
                        }
                    }
                } else {
                    Text("Enter results to track accuracy.")
                        .foregroundColor(.secondary)
                }
            }
            .navigationTitle("Accuracy")
        }
    }
}

// MARK: - Knockout Data Models

struct KnockoutFixture: Codable {
    let team_a: String
    let team_b: String
    let p_a_wins: Double
    let p_b_wins: Double
}

struct TournamentProb: Codable {
    let team: String
    let p_r16: Double
    let p_qf: Double
    let p_sf: Double
    let p_final: Double
    let p_win: Double
}

// MARK: - Knockout Data Models

struct KnockoutMatchProb: Codable {
    let team_a: String
    let team_b: String
    let stadium: String
    let p_a_wins: Double
    let p_b_wins: Double
}

struct TournamentProbData: Codable {
    let team: String
    let p_r16: Double
    let p_qf: Double
    let p_sf: Double
    let p_final: Double
    let p_win: Double
}

struct KnockoutData: Codable {
    let round_of_16_probs: [KnockoutMatchProb]
    let tournament_probs: [TournamentProbData]
}

// MARK: - Knockout View

struct KnockoutView: View {
    @ObservedObject var sim: Simulator

    var body: some View {
        NavigationView {
            List {
                if let data = sim.knockoutData {
                    Section("Round of 16 predictions") {
                        ForEach(data.round_of_16_probs, id: \.team_a) { m in
                            KnockoutMatchRow(match: m)
                        }
                    }

                    Section("Tournament winner odds") {
                        ForEach(data.tournament_probs, id: \.team) { t in
                            HStack {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(t.team)
                                        .font(.subheadline)
                                    HStack(spacing: 8) {
                                        Text("QF \(Int(t.p_qf * 100))%")
                                        Text("SF \(Int(t.p_sf * 100))%")
                                        Text("F \(Int(t.p_final * 100))%")
                                    }
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                                }

                                Spacer()

                                VStack(alignment: .trailing, spacing: 2) {
                                    Text(String(format: "%.1f%%", t.p_win * 100))
                                        .font(.subheadline)
                                        .fontWeight(.medium)
                                        .foregroundColor(winColor(t.p_win))
                                    Text("to win")
                                        .font(.caption2)
                                        .foregroundColor(.secondary)
                                }
                            }
                            .padding(.vertical, 2)
                        }
                    }
                } else {
                    Text("Knockout data not loaded.")
                        .foregroundColor(.secondary)
                }
            }
            .navigationTitle("Knockout stage")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: { sim.loadKnockoutData() }) {
                        Image(systemName: "arrow.clockwise")
                    }
                }
            }
        }
    }

    func winColor(_ prob: Double) -> Color {
        if prob >= 0.15 { return .green }
        if prob >= 0.05 { return .blue }
        return .gray
    }
}

struct KnockoutMatchRow: View {
    let match: KnockoutMatchProb

    var predicted: String {
        match.p_a_wins >= match.p_b_wins ? match.team_a : match.team_b
    }

    var body: some View {
        VStack(spacing: 8) {
            HStack {
                Text(match.team_a)
                    .font(.subheadline)
                    .fontWeight(match.p_a_wins >= match.p_b_wins ? .semibold : .regular)
                    .foregroundColor(match.p_a_wins >= match.p_b_wins ? .primary : .secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)

                Text(match.stadium)
                    .font(.caption2)
                    .foregroundColor(.secondary)
                    .lineLimit(1)

                Text(match.team_b)
                    .font(.subheadline)
                    .fontWeight(match.p_b_wins > match.p_a_wins ? .semibold : .regular)
                    .foregroundColor(match.p_b_wins > match.p_a_wins ? .primary : .secondary)
                    .frame(maxWidth: .infinity, alignment: .trailing)
            }

            GeometryReader { geo in
                HStack(spacing: 2) {
                    Rectangle()
                        .fill(match.p_a_wins >= match.p_b_wins ? Color.green : Color.gray.opacity(0.4))
                        .frame(width: max(2, geo.size.width * match.p_a_wins))
                        .cornerRadius(3)
                    Rectangle()
                        .fill(match.p_b_wins > match.p_a_wins ? Color.blue : Color.gray.opacity(0.4))
                        .frame(width: max(2, geo.size.width * match.p_b_wins))
                        .cornerRadius(3)
                }
            }
            .frame(height: 8)
            .cornerRadius(4)

            HStack {
                Text(String(format: "%.0f%%", match.p_a_wins * 100))
                    .font(.caption2)
                    .foregroundColor(match.p_a_wins >= match.p_b_wins ? .green : .secondary)
                    .fontWeight(.medium)

                Spacer()

                Text("Predicts: \(predicted)")
                    .font(.caption2)
                    .foregroundColor(.secondary)

                Spacer()

                Text(String(format: "%.0f%%", match.p_b_wins * 100))
                    .font(.caption2)
                    .foregroundColor(match.p_b_wins > match.p_a_wins ? .blue : .secondary)
                    .fontWeight(.medium)
            }
        }
        .padding(.vertical, 6)
    }
}

#Preview {
    ContentView()
}
