/**
 * =====================================================================
 * SUNWIN API SERVER v4.2.1
 * TÀI/XỈU PATTERN RECOGNITION ENGINE + API + DASHBOARD
 * =====================================================================
 * 
 * API Endpoints:
 * - GET /api/tx        : Lấy dữ liệu từ API bên ngoài
 * - GET /sun/vilong    : Dự đoán kết quả tiếp theo
 * - GET /dashboard     : Dashboard HTML
 * - GET /history       : Lịch sử dự đoán
 * - GET /export/json   : Xuất dữ liệu JSON
 * - GET /export/csv    : Xuất dữ liệu CSV
 * 
 * Deploy: Render.com
 * =====================================================================
 */

const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');

// Helper Fetch tương thích Node.js 18+ và tránh treo trên Render
async function fetchWithTimeout(url, options = {}, timeout = 8000) {
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeout);
    try {
        const fetchMethod = globalThis.fetch || (await import('node-fetch')).default;
        const response = await fetchMethod(url, {
            ...options,
            signal: controller.signal
        });
        clearTimeout(id);
        return response;
    } catch (error) {
        clearTimeout(id);
        throw error;
    }
}

// =====================================================================
// 1. TÀI/XỈU ENGINE (TỪ thuật toán đã cung cấp)
// =====================================================================

// =====================================================================
// 1. CẤU HÌNH & HẰNG SỐ
// =====================================================================
const CONFIG = {
    MIN_HISTORY: 10,
    MAX_STREAK: 30,
    NGRAM_MAX: 6,
    MARKOV_STATES: 6,
    PATTERN_MIN_SUPPORT: 2,
    CONFIDENCE_THRESHOLD: 0.55,
    WEIGHT_PATTERN: 0.30,
    WEIGHT_TRANSITION: 0.20,
    WEIGHT_MARKOV: 0.15,
    WEIGHT_NGRAM: 0.15,
    WEIGHT_RULE: 0.10,
    WEIGHT_POINT: 0.10
};

// =====================================================================
// 2. CORE DATA STRUCTURES
// =====================================================================
class PatternDatabase {
    constructor() {
        this.db = new Map();
        this.totalEntries = 0;
    }

    add(pattern, next, confidence = 0.5) {
        const key = pattern + '|' + next;
        if (!this.db.has(key)) {
            this.db.set(key, { pattern, next, count: 0, accuracy: 0, support: 0 });
        }
        const entry = this.db.get(key);
        entry.count++;
        entry.support = entry.count;
        entry.accuracy = Math.min(1, entry.accuracy + (confidence - entry.accuracy) / entry.count);
        this.totalEntries++;
    }

    get(pattern, next) {
        const key = pattern + '|' + next;
        return this.db.get(key) || null;
    }

    getTransitions(pattern) {
        const results = [];
        for (const [key, entry] of this.db) {
            if (entry.pattern === pattern) {
                results.push({
                    next: entry.next,
                    count: entry.count,
                    accuracy: entry.accuracy,
                    support: entry.support
                });
            }
        }
        results.sort((a, b) => b.support - a.support);
        return results;
    }

    getTopPatterns(limit = 20) {
        const all = [];
        for (const [key, entry] of this.db) {
            all.push({ 
                pattern: entry.pattern, 
                next: entry.next, 
                count: entry.count, 
                accuracy: entry.accuracy 
            });
        }
        all.sort((a, b) => b.count - a.count);
        return all.slice(0, limit);
    }

    buildFromHistory(history) {
        this.db.clear();
        this.totalEntries = 0;

        for (let len = 1; len <= CONFIG.NGRAM_MAX; len++) {
            for (let i = 0; i <= history.length - len - 1; i++) {
                const pattern = history.slice(i, i + len).join('');
                const next = history[i + len];
                const confidence = this._calculateConfidence(history, pattern, next);
                this.add(pattern, next, confidence);
            }
        }
    }

    _calculateConfidence(history, pattern, next) {
        let totalOccurrences = 0;
        let nextOccurrences = 0;
        const patternLen = pattern.length;

        for (let i = 0; i <= history.length - patternLen - 1; i++) {
            const sub = history.slice(i, i + patternLen).join('');
            if (sub === pattern) {
                totalOccurrences++;
                if (history[i + patternLen] === next) {
                    nextOccurrences++;
                }
            }
        }

        if (totalOccurrences === 0) return 0.5;
        return (nextOccurrences + 1) / (totalOccurrences + 2);
    }
}

// =====================================================================
// 3. MARKOV CHAIN
// =====================================================================
class MarkovChain {
    constructor(order = 2) {
        this.order = Math.min(order, CONFIG.MARKOV_STATES);
        this.transitions = new Map();
        this.totalTransitions = 0;
    }

    train(sequence) {
        this.transitions.clear();
        this.totalTransitions = 0;

        for (let i = 0; i <= sequence.length - this.order - 1; i++) {
            const state = sequence.slice(i, i + this.order).join('');
            const next = sequence[i + this.order];
            
            const key = state + '|' + next;
            if (!this.transitions.has(key)) {
                this.transitions.set(key, { state, next, count: 0 });
            }
            this.transitions.get(key).count++;
            this.totalTransitions++;
        }
    }

    predict(state) {
        const results = [];
        const stateLen = state.length;
        
        const currentState = stateLen >= this.order ? 
            state.slice(stateLen - this.order) : state;

        for (const [key, entry] of this.transitions) {
            if (entry.state === currentState) {
                results.push({
                    next: entry.next,
                    prob: entry.count / this.totalTransitions
                });
            }
        }

        if (results.length === 0 && this.order > 1) {
            const fallback = new MarkovChain(this.order - 1);
            for (const [key, entry] of this.transitions) {
                const stateParts = entry.state.split('');
                const newState = stateParts.slice(stateParts.length - (this.order - 1)).join('');
                const newKey = newState + '|' + entry.next;
                if (!fallback.transitions.has(newKey)) {
                    fallback.transitions.set(newKey, { state: newState, next: entry.next, count: 0 });
                }
                fallback.transitions.get(newKey).count += entry.count;
                fallback.totalTransitions += entry.count;
            }
            return fallback.predict(currentState.slice(1));
        }

        return results;
    }

    getProbability(state, next) {
        const key = state + '|' + next;
        const entry = this.transitions.get(key);
        if (!entry) return 0;
        return entry.count / this.totalTransitions;
    }
}

// =====================================================================
// 4. N-GRAM ANALYZER
// =====================================================================
class NGramAnalyzer {
    constructor(maxN = CONFIG.NGRAM_MAX) {
        this.maxN = maxN;
        this.ngrams = new Map();
        this.totalNGrams = 0;
    }

    build(sequence) {
        this.ngrams.clear();
        this.totalNGrams = 0;

        for (let n = 2; n <= this.maxN; n++) {
            for (let i = 0; i <= sequence.length - n; i++) {
                const gram = sequence.slice(i, i + n).join('');
                if (!this.ngrams.has(gram)) {
                    this.ngrams.set(gram, { count: 0, next: {} });
                }
                const entry = this.ngrams.get(gram);
                entry.count++;
                this.totalNGrams++;

                if (i + n < sequence.length) {
                    const next = sequence[i + n];
                    entry.next[next] = (entry.next[next] || 0) + 1;
                }
            }
        }
    }

    analyze(gram) {
        const entry = this.ngrams.get(gram);
        if (!entry) return null;

        const totalNext = Object.values(entry.next).reduce((a, b) => a + b, 0);
        const nextProbs = {};
        for (const [next, count] of Object.entries(entry.next)) {
            nextProbs[next] = count / totalNext;
        }

        return {
            count: entry.count,
            frequency: entry.count / this.totalNGrams,
            next: nextProbs,
            support: entry.count
        };
    }

    getTopNGrams(limit = 20) {
        const all = [];
        for (const [gram, entry] of this.ngrams) {
            all.push({ gram, count: entry.count });
        }
        all.sort((a, b) => b.count - a.count);
        return all.slice(0, limit);
    }
}

// =====================================================================
// 5. STREAK ANALYZER
// =====================================================================
class StreakAnalyzer {
    analyze(history) {
        if (!history || history.length === 0) {
            return { streaks: [], maxStreak: 0, currentStreak: 0, currentResult: null };
        }

        const streaks = [];
        let currentStreak = 1;
        let currentResult = history[0];
        let maxStreak = 0;

        for (let i = 1; i < history.length; i++) {
            if (history[i] === currentResult) {
                currentStreak++;
            } else {
                streaks.push({ result: currentResult, length: currentStreak });
                maxStreak = Math.max(maxStreak, currentStreak);
                currentResult = history[i];
                currentStreak = 1;
            }
        }
        streaks.push({ result: currentResult, length: currentStreak });
        maxStreak = Math.max(maxStreak, currentStreak);

        const streakStats = {
            byLength: {},
            byResult: { T: {}, X: {} }
        };

        for (const streak of streaks) {
            const len = streak.length;
            streakStats.byLength[len] = (streakStats.byLength[len] || 0) + 1;
            streakStats.byResult[streak.result][len] = (streakStats.byResult[streak.result][len] || 0) + 1;
        }

        return {
            streaks,
            maxStreak,
            currentStreak: streaks.length > 0 ? streaks[streaks.length - 1].length : 0,
            currentResult: streaks.length > 0 ? streaks[streaks.length - 1].result : null,
            stats: streakStats,
            totalStreaks: streaks.length
        };
    }

    predictBreak(history, currentResult, streakLength) {
        let totalLongStreaks = 0;
        let brokenStreaks = 0;

        for (let i = 0; i < history.length - 1; i++) {
            let count = 1;
            while (i + count < history.length && history[i + count] === history[i]) {
                count++;
            }
            if (count >= streakLength) {
                totalLongStreaks++;
                if (i + count < history.length && history[i + count] !== history[i]) {
                    brokenStreaks++;
                }
            }
            i += count - 1;
        }

        const breakProb = totalLongStreaks > 0 ? brokenStreaks / totalLongStreaks : 0.5;
        const continueProb = 1 - breakProb;

        return {
            breakProbability: breakProb,
            continueProbability: continueProb,
            confidence: Math.min(1, totalLongStreaks / 10)
        };
    }
}

// =====================================================================
// 6. RULE ENGINE
// =====================================================================
class RuleEngine {
    constructor() {
        this.rules = [];
        this.loadedRules = new Set();
    }

    addRule(rule) {
        const key = rule.pattern + '|' + rule.next;
        if (!this.loadedRules.has(key)) {
            this.rules.push(rule);
            this.loadedRules.add(key);
        }
    }

    loadRules(rulesData) {
        for (const rule of rulesData) {
            this.addRule(rule);
        }
    }

    apply(currentPattern) {
        const results = [];
        for (const rule of this.rules) {
            if (currentPattern.endsWith(rule.pattern) || currentPattern === rule.pattern) {
                results.push({
                    next: rule.next,
                    confidence: rule.confidence || 0.6,
                    rule: rule
                });
            }
        }
        results.sort((a, b) => b.confidence - a.confidence);
        return results;
    }

    static parseRulesFromText(rawContent) {
        const rules = [];
        const lines = rawContent.split('\n');
        
        for (const line of lines) {
            const trimmed = line.trim();
            const match = trimmed.match(/if\s*\(.*?['"]([TX]+)['"].*?return\s*['"]([TX])['"]/i);
            if (match) {
                rules.push({
                    pattern: match[1],
                    next: match[2],
                    confidence: 0.7,
                    source: 'thuattoan.txt'
                });
            }
            
            const caseMatch = trimmed.match(/case\s*['"]([TX]+)['"]\s*:\s*return\s*['"]([TX])['"]/i);
            if (caseMatch) {
                rules.push({
                    pattern: caseMatch[1],
                    next: caseMatch[2],
                    confidence: 0.65,
                    source: 'thuattoan.txt'
                });
            }
        }
        
        return rules;
    }
}

// =====================================================================
// 7. POINT ANALYZER
// =====================================================================
class PointAnalyzer {
    analyze(history) {
        const pointStats = {};
        const pointTransitions = {};
        let total = 0;

        for (let i = 0; i < history.length; i++) {
            const point = history[i].tong || history[i].total || 0;
            if (point < 3 || point > 18) continue;
            
            total++;
            pointStats[point] = (pointStats[point] || 0) + 1;
            
            if (i < history.length - 1) {
                const nextPoint = history[i + 1].tong || history[i + 1].total || 0;
                if (nextPoint >= 3 && nextPoint <= 18) {
                    const key = point + '|' + nextPoint;
                    pointTransitions[key] = (pointTransitions[key] || 0) + 1;
                }
            }
        }

        const frequency = {};
        for (const [point, count] of Object.entries(pointStats)) {
            frequency[point] = count / total;
        }

        let increasing = 0;
        let decreasing = 0;
        let stable = 0;

        for (let i = 1; i < history.length; i++) {
            const prev = history[i-1].tong || history[i-1].total || 0;
            const curr = history[i].tong || history[i].total || 0;
            if (prev >= 3 && prev <= 18 && curr >= 3 && curr <= 18) {
                if (curr > prev) increasing++;
                else if (curr < prev) decreasing++;
                else stable++;
            }
        }

        const trend = increasing > decreasing ? 'up' : (decreasing > increasing ? 'down' : 'stable');

        return {
            pointStats,
            pointTransitions,
            frequency,
            total,
            trend,
            trendRatio: {
                increasing: increasing / (increasing + decreasing + stable || 1),
                decreasing: decreasing / (increasing + decreasing + stable || 1),
                stable: stable / (increasing + decreasing + stable || 1)
            }
        };
    }

    predictFromPoints(recentPoints, pointStats) {
        if (recentPoints.length < 3) {
            return { prediction: null, confidence: 0 };
        }

        const avg = recentPoints.reduce((a, b) => a + b, 0) / recentPoints.length;
        
        let taiThreshold = 11;
        let xiuThreshold = 10;

        if (pointStats) {
            const points = Object.keys(pointStats).map(Number);
            if (points.length > 0) {
                const sorted = points.sort((a, b) => a - b);
                const median = sorted[Math.floor(sorted.length / 2)];
                taiThreshold = Math.round(median + 1);
                xiuThreshold = Math.round(median - 1);
            }
        }

        const prediction = avg >= taiThreshold ? 'T' : (avg <= xiuThreshold ? 'X' : null);
        const confidence = Math.min(0.9, Math.abs(avg - 10.5) / 5);

        return { prediction, confidence, avgPoint: avg };
    }
}

// =====================================================================
// 8. MAIN ENGINE
// =====================================================================
class TaiXiuEngine {
    constructor() {
        this.history = [];
        this.rawData = [];
        this.patternDB = new PatternDatabase();
        this.markovChain = new MarkovChain(2);
        this.ngramAnalyzer = new NGramAnalyzer(CONFIG.NGRAM_MAX);
        this.streakAnalyzer = new StreakAnalyzer();
        this.ruleEngine = new RuleEngine();
        this.pointAnalyzer = new PointAnalyzer();
        this.isInitialized = false;
    }

    initialize(data) {
        this.rawData = data;
        this.history = this._extractResults(data);
        this._buildAllModels();
        this.isInitialized = true;
        return this;
    }

    _extractResults(data) {
        const results = [];
        for (const item of data) {
            if (item.ket_qua === 'Tài' || item.ket_qua === 'Xỉu') {
                results.push(item.ket_qua === 'Tài' ? 'T' : 'X');
            } else if (item.result === 'Tài' || item.result === 'Xỉu') {
                results.push(item.result === 'Tài' ? 'T' : 'X');
            } else if (typeof item === 'string') {
                if (item === 'T' || item === 'X') results.push(item);
            }
        }
        return results;
    }

    _buildAllModels() {
        if (this.history.length < CONFIG.MIN_HISTORY) {
            console.warn('Lịch sử quá ngắn để phân tích đáng tin cậy');
            return;
        }

        this.patternDB.buildFromHistory(this.history);
        this.markovChain.train(this.history);
        this.ngramAnalyzer.build(this.history);
    }

    loadRules(rawContent) {
        const rules = RuleEngine.parseRulesFromText(rawContent);
        this.ruleEngine.loadRules(rules);
        return this;
    }

    analyze(lookback = 10) {
        if (!this.isInitialized || this.history.length < CONFIG.MIN_HISTORY) {
            return {
                prediction: null,
                confidence: 0,
                reason: 'Không đủ dữ liệu để phân tích',
                scores: {},
                matchedPatterns: [],
                matchedRules: []
            };
        }

        const recentHistory = this.history.slice(-lookback);
        const currentPattern = recentHistory.join('');
        const lastResult = this.history[this.history.length - 1];
        const streakInfo = this.streakAnalyzer.analyze(this.history);
        const currentStreak = streakInfo.currentStreak;
        const currentResult = streakInfo.currentResult;

        const results = {
            pattern: this._analyzePattern(currentPattern),
            transition: this._analyzeTransition(currentPattern, lastResult),
            markov: this._analyzeMarkov(currentPattern),
            ngram: this._analyzeNGram(currentPattern),
            streak: this._analyzeStreak(currentResult, currentStreak, recentHistory),
            point: this._analyzePoint(),
            rule: this._analyzeRules(currentPattern)
        };

        const scores = this._calculateScores(results);
        const finalPrediction = this._getFinalPrediction(scores);
        const confidence = this._calculateConfidence(scores);
        const reason = this._generateReason(results, finalPrediction);
        const matchedPatterns = this._getMatchedPatterns(currentPattern);

        return {
            prediction: finalPrediction,
            confidence: confidence,
            reason: reason,
            scores: scores,
            matchedPatterns: matchedPatterns,
            matchedRules: results.rule.matchedRules || [],
            details: {
                currentStreak,
                currentResult,
                totalHistory: this.history.length,
                patternDB: this.patternDB.getTopPatterns(5),
                topNGrams: this.ngramAnalyzer.getTopNGrams(5)
            }
        };
    }

    _analyzePattern(currentPattern) {
        const predictions = [];
        for (let len = Math.min(6, currentPattern.length); len >= 1; len--) {
            const subPattern = currentPattern.slice(currentPattern.length - len);
            const transitions = this.patternDB.getTransitions(subPattern);
            if (transitions.length > 0) {
                for (const t of transitions) {
                    predictions.push({
                        next: t.next,
                        confidence: t.accuracy,
                        support: t.support,
                        pattern: subPattern,
                        weight: 0.3
                    });
                }
            }
        }
        return this._aggregatePredictions(predictions);
    }

    _analyzeTransition(currentPattern, lastResult) {
        const predictions = [];
        
        let totalT = 0, totalX = 0;
        for (let i = 0; i < this.history.length - 1; i++) {
            if (this.history[i] === lastResult) {
                if (this.history[i + 1] === 'T') totalT++;
                else totalX++;
            }
        }

        const total = totalT + totalX;
        if (total > 0) {
            predictions.push(
                { next: 'T', confidence: totalT / total, weight: 0.2 },
                { next: 'X', confidence: totalX / total, weight: 0.2 }
            );
        }

        return this._aggregatePredictions(predictions);
    }

    _analyzeMarkov(currentPattern) {
        const predictions = [];
        const state = currentPattern.slice(-this.markovChain.order);
        const results = this.markovChain.predict(state);
        
        for (const r of results) {
            predictions.push({
                next: r.next,
                confidence: r.prob,
                weight: 0.15
            });
        }

        return this._aggregatePredictions(predictions);
    }

    _analyzeNGram(currentPattern) {
        const predictions = [];
        
        for (let n = 2; n <= CONFIG.NGRAM_MAX; n++) {
            if (currentPattern.length >= n) {
                const gram = currentPattern.slice(-n);
                const analysis = this.ngramAnalyzer.analyze(gram);
                if (analysis && analysis.next) {
                    for (const [next, prob] of Object.entries(analysis.next)) {
                        predictions.push({
                            next: next,
                            confidence: prob,
                            support: analysis.count,
                            weight: 0.15
                        });
                    }
                }
            }
        }

        return this._aggregatePredictions(predictions);
    }

    _analyzeStreak(currentResult, streakLength, recentHistory) {
        const predictions = [];
        
        if (streakLength >= 3) {
            const breakInfo = this.streakAnalyzer.predictBreak(
                this.history, 
                currentResult, 
                streakLength
            );
            
            const nextResult = currentResult === 'T' ? 'X' : 'T';
            predictions.push({
                next: nextResult,
                confidence: breakInfo.breakProbability,
                weight: 0.1
            });
            predictions.push({
                next: currentResult,
                confidence: breakInfo.continueProbability,
                weight: 0.1
            });
        }

        return this._aggregatePredictions(predictions);
    }

    _analyzePoint() {
        const predictions = [];
        
        if (this.rawData && this.rawData.length > 0) {
            const recentPoints = this.rawData.slice(-5).map(item => item.tong || item.total || 0)
                .filter(p => p >= 3 && p <= 18);
            
            if (recentPoints.length >= 3) {
                const pointStats = this.pointAnalyzer.analyze(this.rawData);
                const result = this.pointAnalyzer.predictFromPoints(recentPoints, pointStats.pointStats);
                
                if (result.prediction) {
                    predictions.push({
                        next: result.prediction,
                        confidence: result.confidence,
                        weight: 0.1
                    });
                }
            }
        }

        return this._aggregatePredictions(predictions);
    }

    _analyzeRules(currentPattern) {
        const matchedRules = this.ruleEngine.apply(currentPattern);
        const predictions = [];
        
        for (const r of matchedRules) {
            predictions.push({
                next: r.next,
                confidence: r.confidence,
                weight: 0.1,
                rule: r.rule
            });
        }

        return {
            aggregated: this._aggregatePredictions(predictions),
            matchedRules: matchedRules
        };
    }

    _aggregatePredictions(predictions) {
        if (predictions.length === 0) {
            return { T: 0, X: 0, total: 0 };
        }

        let scoreT = 0, scoreX = 0;
        let totalWeight = 0;

        for (const p of predictions) {
            const weight = p.weight || 0.1;
            const confidence = p.confidence || 0.5;
            if (p.next === 'T') scoreT += confidence * weight;
            else if (p.next === 'X') scoreX += confidence * weight;
            totalWeight += weight;
        }

        return {
            T: totalWeight > 0 ? scoreT / totalWeight : 0,
            X: totalWeight > 0 ? scoreX / totalWeight : 0,
            total: totalWeight
        };
    }

    _calculateScores(results) {
        const weights = CONFIG;
        let scoreT = 0, scoreX = 0;
        let totalWeight = 0;

        if (results.pattern) {
            scoreT += results.pattern.T * weights.WEIGHT_PATTERN;
            scoreX += results.pattern.X * weights.WEIGHT_PATTERN;
            totalWeight += weights.WEIGHT_PATTERN;
        }

        if (results.transition) {
            scoreT += results.transition.T * weights.WEIGHT_TRANSITION;
            scoreX += results.transition.X * weights.WEIGHT_TRANSITION;
            totalWeight += weights.WEIGHT_TRANSITION;
        }

        if (results.markov) {
            scoreT += results.markov.T * weights.WEIGHT_MARKOV;
            scoreX += results.markov.X * weights.WEIGHT_MARKOV;
            totalWeight += weights.WEIGHT_MARKOV;
        }

        if (results.ngram) {
            scoreT += results.ngram.T * weights.WEIGHT_NGRAM;
            scoreX += results.ngram.X * weights.WEIGHT_NGRAM;
            totalWeight += weights.WEIGHT_NGRAM;
        }

        if (results.streak) {
            scoreT += results.streak.T * weights.WEIGHT_PATTERN * 0.5;
            scoreX += results.streak.X * weights.WEIGHT_PATTERN * 0.5;
            totalWeight += weights.WEIGHT_PATTERN * 0.5;
        }

        if (results.point) {
            scoreT += results.point.T * weights.WEIGHT_POINT;
            scoreX += results.point.X * weights.WEIGHT_POINT;
            totalWeight += weights.WEIGHT_POINT;
        }

        if (results.rule && results.rule.aggregated) {
            scoreT += results.rule.aggregated.T * weights.WEIGHT_RULE;
            scoreX += results.rule.aggregated.X * weights.WEIGHT_RULE;
            totalWeight += weights.WEIGHT_RULE;
        }

        const finalT = totalWeight > 0 ? scoreT / totalWeight : 0.5;
        const finalX = totalWeight > 0 ? scoreX / totalWeight : 0.5;
        const sum = finalT + finalX;
        
        return {
            T: sum > 0 ? finalT / sum : 0.5,
            X: sum > 0 ? finalX / sum : 0.5,
            raw: { scoreT, scoreX, totalWeight }
        };
    }

    _getFinalPrediction(scores) {
        if (scores.T > scores.X && scores.T > CONFIG.CONFIDENCE_THRESHOLD) {
            return 'T';
        } else if (scores.X > scores.T && scores.X > CONFIG.CONFIDENCE_THRESHOLD) {
            return 'X';
        }
        return null;
    }

    _calculateConfidence(scores) {
        const diff = Math.abs(scores.T - scores.X);
        return Math.min(0.95, 0.5 + diff * 0.4);
    }

    _generateReason(results, prediction) {
        const reasons = [];
        
        if (results.pattern && results.pattern.T > 0.6) {
            reasons.push(`Pattern Database nghiêng về Tài (${Math.round(results.pattern.T * 100)}%)`);
        } else if (results.pattern && results.pattern.X > 0.6) {
            reasons.push(`Pattern Database nghiêng về Xỉu (${Math.round(results.pattern.X * 100)}%)`);
        }

        if (results.transition) {
            const max = results.transition.T > results.transition.X ? 'T' : 'X';
            const prob = Math.max(results.transition.T, results.transition.X);
            if (prob > 0.55) {
                reasons.push(`Chuyển tiếp ${max} với xác suất ${Math.round(prob * 100)}%`);
            }
        }

        if (results.markov && results.markov.T > 0.55) {
            reasons.push(`Markov Chain dự đoán Tài (${Math.round(results.markov.T * 100)}%)`);
        } else if (results.markov && results.markov.X > 0.55) {
            reasons.push(`Markov Chain dự đoán Xỉu (${Math.round(results.markov.X * 100)}%)`);
        }

        if (results.streak) {
            if (results.streak.T > 0.6) {
                reasons.push(`Phân tích chuỗi cho thấy khả năng tiếp tục Tài`);
            } else if (results.streak.X > 0.6) {
                reasons.push(`Phân tích chuỗi cho thấy khả năng tiếp tục Xỉu`);
            }
        }

        if (results.rule && results.rule.matchedRules.length > 0) {
            const bestRule = results.rule.matchedRules[0];
            reasons.push(`Rule phù hợp: ${bestRule.rule.pattern} → ${bestRule.next} (độ tin cậy ${Math.round(bestRule.confidence * 100)}%)`);
        }

        if (reasons.length === 0) {
            reasons.push('Dữ liệu không đủ rõ ràng để đưa ra dự đoán chắc chắn');
        }

        return reasons.join('. ');
    }

    _getMatchedPatterns(currentPattern) {
        const matched = [];
        for (let len = Math.min(6, currentPattern.length); len >= 1; len--) {
            const sub = currentPattern.slice(currentPattern.length - len);
            const transitions = this.patternDB.getTransitions(sub);
            if (transitions.length > 0) {
                matched.push({
                    pattern: sub,
                    transitions: transitions.map(t => ({
                        next: t.next,
                        count: t.count,
                        accuracy: Math.round(t.accuracy * 100) + '%'
                    }))
                });
            }
        }
        return matched;
    }
}

// =====================================================================
// 9. EXPRESS SERVER
// =====================================================================

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// =====================================================================
// 10. DATA STORAGE
// =====================================================================

const DATA_FILE = path.join(__dirname, 'data', 'history.json');
const PREDICTIONS_FILE = path.join(__dirname, 'data', 'predictions.json');

// Safe file write helper
function safeWriteFileSync(filePath, content) {
    try {
        const dir = path.dirname(filePath);
        if (!fs.existsSync(dir)) {
            fs.mkdirSync(dir, { recursive: true });
        }
        fs.writeFileSync(filePath, content, 'utf8');
    } catch (e) {
        console.error(`Error writing file ${filePath}:`, e.message);
    }
}

// Ensure data directory exists
if (!fs.existsSync(path.join(__dirname, 'data'))) {
    try {
        fs.mkdirSync(path.join(__dirname, 'data'), { recursive: true });
    } catch (e) {
        console.error('Error creating data dir:', e);
    }
}

// Load saved data
let savedData = [];
let predictionHistory = [];

try {
    if (fs.existsSync(DATA_FILE)) {
        savedData = JSON.parse(fs.readFileSync(DATA_FILE, 'utf8'));
    }
} catch (e) {
    console.error('Error loading data:', e);
}

try {
    if (fs.existsSync(PREDICTIONS_FILE)) {
        predictionHistory = JSON.parse(fs.readFileSync(PREDICTIONS_FILE, 'utf8'));
    }
} catch (e) {
    console.error('Error loading predictions:', e);
}

// =====================================================================
// 11. API ENDPOINTS
// =====================================================================

// Lấy dữ liệu từ API bên ngoài
app.get('/api/tx', async (req, res) => {
    try {
        const response = await fetchWithTimeout('http://160.191.244.75:9012/api/tx', {}, 8000);
        const data = await response.json();
        
        // Lưu dữ liệu
        if (Array.isArray(data) && data.length > 0) {
            savedData = data;
            safeWriteFileSync(DATA_FILE, JSON.stringify(data, null, 2));
        }
        
        res.json(data);
    } catch (error) {
        console.error('Error fetching data:', error.message);
        // Trả về dữ liệu đã lưu nếu có
        if (savedData.length > 0) {
            res.json(savedData);
        } else {
            res.status(500).json({ error: 'Không thể lấy dữ liệu' });
        }
    }
});

// API dự đoán (ĐÃ ĐỔI LINK THÀNH /sun/vilong VÀ CẬP NHẬT ID)
app.get('/sun/vilong', async (req, res) => {
    try {
        // Lấy dữ liệu mới nhất
        try {
            const response = await fetchWithTimeout('http://160.191.244.75:9012/api/tx', {}, 8000);
            const data = await response.json();
            
            if (Array.isArray(data) && data.length > 0) {
                savedData = data;
                safeWriteFileSync(DATA_FILE, JSON.stringify(data, null, 2));
            }
        } catch (fetchErr) {
            console.warn('Cannot update external data, using cached savedData:', fetchErr.message);
        }

        if (savedData.length < 10) {
            return res.json({
                phien_truoc: null,
                ket_qua: null,
                phien_hien_tai: null,
                du_doan: null,
                do_tin_cay: 0,
                giai_thich: 'Không đủ dữ liệu để phân tích',
                idol: '@cskhvilong1'
            });
        }

        // Lấy phiên hiện tại và phiên trước
        const currentSession = savedData[0];
        const prevSession = savedData[1] || savedData[0];

        // Khởi tạo engine và phân tích
        const engine = new TaiXiuEngine();
        engine.initialize(savedData);
        
        // Nạp rules từ thuattoan.txt nếu có
        const thuattoanPath = path.join(__dirname, 'thuattoan.txt');
        if (fs.existsSync(thuattoanPath)) {
            const rulesContent = fs.readFileSync(thuattoanPath, 'utf8');
            engine.loadRules(rulesContent);
        }

        const result = engine.analyze(10);
        
        // Chuyển đổi dự đoán sang tiếng Việt
        const duDoan = result.prediction === 'T' ? 'Tài' : (result.prediction === 'X' ? 'Xỉu' : 'Chưa xác định');
        const doTinCay = Math.round(result.confidence * 100);
        const giaiThich = result.reason || 'Phân tích từ dữ liệu lịch sử';

        // Lưu dự đoán
        const predictionRecord = {
            phien: currentSession.phien || currentSession.Phien || 0,
            phien_truoc: prevSession.phien || prevSession.Phien || 0,
            ket_qua: currentSession.ket_qua || currentSession.KetQua || '',
            du_doan: duDoan,
            do_tin_cay: doTinCay,
            giai_thich: giaiThich,
            timestamp: new Date().toISOString(),
            idol: '@cskhvilong1'
        };

        predictionHistory.push(predictionRecord);
        if (predictionHistory.length > 1000) {
            predictionHistory = predictionHistory.slice(-1000);
        }
        safeWriteFileSync(PREDICTIONS_FILE, JSON.stringify(predictionHistory, null, 2));

        res.json({
            phien_truoc: prevSession.phien || prevSession.Phien || 0,
            ket_qua: prevSession.ket_qua || prevSession.KetQua || '',
            phien_hien_tai: currentSession.phien || currentSession.Phien || 0,
            du_doan: duDoan,
            do_tin_cay: doTinCay,
            giai_thich: giaiThich,
            idol: '@cskhvilong1',
            raw: result
        });

    } catch (error) {
        console.error('Prediction error:', error);
        res.status(500).json({
            error: 'Lỗi khi dự đoán',
            message: error.message
        });
    }
});

// Lấy lịch sử dự đoán
app.get('/history', (req, res) => {
    const limit = parseInt(req.query.limit) || 100;
    res.json(predictionHistory.slice(-limit));
});

// Xuất dữ liệu JSON
app.get('/export/json', (req, res) => {
    const data = {
        predictions: predictionHistory,
        raw_data: savedData,
        exported_at: new Date().toISOString()
    };
    res.setHeader('Content-Type', 'application/json');
    res.setHeader('Content-Disposition', 'attachment; filename=dulieu.json');
    res.json(data);
});

// Xuất dữ liệu CSV
app.get('/export/csv', (req, res) => {
    const headers = ['phien', 'ket_qua', 'du_doan', 'do_tin_cay', 'giai_thich', 'timestamp'];
    let csv = headers.join(',') + '\n';
    
    for (const record of predictionHistory) {
        const row = headers.map(h => {
            let val = record[h] || '';
            if (typeof val === 'string' && val.includes(',')) {
                val = '"' + val + '"';
            }
            return val;
        });
        csv += row.join(',') + '\n';
    }
    
    res.setHeader('Content-Type', 'text/csv');
    res.setHeader('Content-Disposition', 'attachment; filename=dulieu.csv');
    res.send(csv);
});

// Dashboard HTML
app.get('/dashboard', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'dashboard.html'));
});

// =====================================================================
// 12. SERVE STATIC FILES
// =====================================================================

// Tạo thư mục public nếu chưa có
if (!fs.existsSync(path.join(__dirname, 'public'))) {
    try {
        fs.mkdirSync(path.join(__dirname, 'public'), { recursive: true });
    } catch (e) {
        console.error('Error creating public dir:', e);
    }
}

// Tạo dashboard HTML
const dashboardHTML = `<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sunwin Dashboard - Dự Đoán Tài Xỉu</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        :root {
            --bg: #0a0e17;
            --bg-card: rgba(255,255,255,0.05);
            --text: #e8edf5;
            --text-secondary: #8899bb;
            --border: rgba(255,255,255,0.08);
            --primary: #2563eb;
            --primary-glow: rgba(37,99,235,0.3);
            --success: #22c55e;
            --danger: #dc2626;
            --warning: #f59e0b;
            --radius: 16px;
            --shadow: 0 8px 32px rgba(0,0,0,0.4);
        }
        [data-theme="light"] {
            --bg: #f0f2f5;
            --bg-card: rgba(255,255,255,0.8);
            --text: #1a1a2e;
            --text-secondary: #4a4a6a;
            --border: rgba(0,0,0,0.1);
        }
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg);
            color: var(--text);
            transition: all 0.3s ease;
            padding: 20px;
            min-height: 100vh;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        
        /* Glassmorphism */
        .glass {
            background: var(--bg-card);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            padding: 24px;
            transition: all 0.3s ease;
        }
        .glass:hover { transform: translateY(-2px); }
        
        /* Header */
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 16px 24px;
            margin-bottom: 30px;
            border-radius: var(--radius);
            background: var(--bg-card);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border);
        }
        .header h1 {
            font-size: 24px;
            font-weight: 700;
            background: linear-gradient(135deg, #2563eb, #60a5fa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .header-controls { display: flex; gap: 12px; align-items: center; }
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: var(--radius);
            font-weight: 600;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.2s ease;
            background: var(--bg-card);
            color: var(--text);
            border: 1px solid var(--border);
        }
        .btn:hover { transform: scale(1.02); }
        .btn-primary {
            background: linear-gradient(135deg, #2563eb, #3b82f6);
            color: white;
            border: none;
        }
        .btn-primary:hover { background: linear-gradient(135deg, #1d4ed8, #2563eb); }
        .btn-success {
            background: linear-gradient(135deg, #22c55e, #4ade80);
            color: white;
            border: none;
        }
        .btn-danger {
            background: linear-gradient(135deg, #dc2626, #ef4444);
            color: white;
            border: none;
        }
        .btn-sm { padding: 6px 14px; font-size: 12px; }
        
        /* Grid */
        .grid-3 { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px; }
        
        /* Stats */
        .stat-card { text-align: center; padding: 20px; }
        .stat-number {
            font-size: 42px;
            font-weight: 700;
            background: linear-gradient(135deg, #2563eb, #60a5fa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .stat-label { font-size: 14px; color: var(--text-secondary); margin-top: 4px; text-transform: uppercase; letter-spacing: 1px; }
        .stat-success .stat-number {
            background: linear-gradient(135deg, #22c55e, #4ade80);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .stat-danger .stat-number {
            background: linear-gradient(135deg, #dc2626, #f87171);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .stat-warning .stat-number {
            background: linear-gradient(135deg, #f59e0b, #fbbf24);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        /* Prediction Card */
        .prediction-result {
            text-align: center;
            padding: 30px;
        }
        .prediction-label {
            font-size: 16px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        .prediction-value {
            font-size: 56px;
            font-weight: 800;
            margin: 10px 0;
        }
        .prediction-value.tai { color: #2563eb; }
        .prediction-value.xiu { color: #dc2626; }
        .prediction-confidence {
            font-size: 18px;
            color: var(--text-secondary);
        }
        .confidence-bar {
            width: 100%;
            height: 8px;
            background: var(--border);
            border-radius: 4px;
            margin-top: 12px;
            overflow: hidden;
        }
        .confidence-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.5s ease;
            background: linear-gradient(90deg, #2563eb, #60a5fa);
        }
        .prediction-reason {
            margin-top: 16px;
            padding: 12px 16px;
            background: rgba(37,99,235,0.1);
            border-radius: var(--radius);
            font-size: 14px;
            color: var(--text-secondary);
            text-align: left;
        }
        
        /* Table */
        .table-container { max-height: 400px; overflow-y: auto; }
        .table-container::-webkit-scrollbar { width: 6px; }
        .table-container::-webkit-scrollbar-track { background: transparent; }
        .table-container::-webkit-scrollbar-thumb { background: var(--primary); border-radius: 3px; }
        table { width: 100%; border-collapse: collapse; font-size: 14px; }
        th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border); }
        th { color: var(--text-secondary); font-weight: 600; position: sticky; top: 0; background: var(--bg); }
        .status-correct { color: var(--success); }
        .status-wrong { color: var(--danger); }
        .status-pending { color: var(--warning); }
        
        /* Chart */
        .chart-container { height: 200px; display: flex; align-items: flex-end; gap: 4px; padding-top: 20px; }
        .chart-bar {
            flex: 1;
            border-radius: 4px 4px 0 0;
            min-height: 4px;
            transition: height 0.5s ease;
            position: relative;
        }
        .chart-bar.tai { background: linear-gradient(180deg, #2563eb, #60a5fa); }
        .chart-bar.xiu { background: linear-gradient(180deg, #dc2626, #f87171); }
        .chart-bar-label {
            position: absolute;
            top: -20px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 10px;
            color: var(--text-secondary);
        }
        
        /* Pattern list */
        .pattern-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 12px;
            border-bottom: 1px solid var(--border);
        }
        .pattern-item:last-child { border-bottom: none; }
        .pattern-text { font-weight: 600; font-size: 15px; }
        .pattern-next { padding: 2px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }
        .pattern-next.tai { background: rgba(37,99,235,0.2); color: #2563eb; }
        .pattern-next.xiu { background: rgba(220,38,38,0.2); color: #dc2626; }
        .pattern-accuracy { font-size: 12px; color: var(--text-secondary); }
        
        /* Responsive */
        @media (max-width: 768px) {
            .grid-2 { grid-template-columns: 1fr; }
            .header { flex-direction: column; gap: 12px; text-align: center; }
            .header-controls { flex-wrap: wrap; justify-content: center; }
            .stat-number { font-size: 28px; }
            .prediction-value { font-size: 36px; }
        }
        @media (max-width: 480px) {
            body { padding: 10px; }
            .glass { padding: 16px; }
            .grid-3 { grid-template-columns: 1fr; }
        }
        
        /* Loading */
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid var(--border);
            border-top-color: var(--primary);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        
        .toggle {
            position: relative;
            width: 48px;
            height: 26px;
            background: var(--border);
            border-radius: 13px;
            cursor: pointer;
            transition: background 0.3s;
            flex-shrink: 0;
        }
        .toggle.active { background: #2563eb; }
        .toggle::after {
            content: '';
            position: absolute;
            top: 3px;
            left: 3px;
            width: 20px;
            height: 20px;
            background: white;
            border-radius: 50%;
            transition: transform 0.3s;
        }
        .toggle.active::after { transform: translateX(22px); }
        
        .auto-refresh { display: flex; align-items: center; gap: 8px; }
        .auto-refresh input[type="checkbox"] { width: 18px; height: 18px; accent-color: var(--primary); }
        .auto-refresh label { font-size: 13px; color: var(--text-secondary); cursor: pointer; }
        
        /* Scrollbar global */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: var(--bg); }
        ::-webkit-scrollbar-thumb { background: var(--primary); border-radius: 4px; }
    </style>
</head>
<body>
<div class="container">
    
    <!-- Header -->
    <header class="header">
        <h1>🎲 Sunwin Dashboard · Dự Đoán Tài Xỉu</h1>
        <div class="header-controls">
            <div class="auto-refresh">
                <input type="checkbox" id="autoRefresh" checked>
                <label for="autoRefresh">Tự động cập nhật</label>
            </div>
            <button class="btn btn-sm" onclick="toggleTheme()">🌓</button>
            <button class="btn btn-sm btn-primary" onclick="fetchPrediction()">🔄 Cập nhật</button>
            <button class="btn btn-sm btn-success" onclick="exportJSON()">📥 JSON</button>
            <button class="btn btn-sm btn-success" onclick="exportCSV()">📥 CSV</button>
        </div>
    </header>
    
    <!-- Prediction Card -->
    <div class="glass" style="margin-bottom:30px;">
        <div class="prediction-result">
            <div class="prediction-label">📊 Dự Đoán Phiên Tiếp Theo</div>
            <div class="prediction-value" id="predictionValue">---</div>
            <div class="prediction-confidence">Độ tin cậy: <span id="confidenceValue">0%</span></div>
            <div class="confidence-bar">
                <div class="confidence-fill" id="confidenceFill" style="width:0%;"></div>
            </div>
            <div class="prediction-reason" id="reasonText">
                Đang tải dữ liệu phân tích...
            </div>
            <div style="margin-top:12px; display:flex; justify-content:center; gap:20px; flex-wrap:wrap;">
                <span style="font-size:13px; color:var(--text-secondary);">Phiên hiện tại: <strong id="currentPhien">---</strong></span>
                <span style="font-size:13px; color:var(--text-secondary);">Kết quả trước: <strong id="prevResult">---</strong></span>
            </div>
        </div>
    </div>
    
    <!-- Stats -->
    <div class="grid-3" id="statsGrid">
        <div class="glass stat-card">
            <div class="stat-number" id="totalPredictions">0</div>
            <div class="stat-label">Tổng Dự Đoán</div>
        </div>
        <div class="glass stat-card stat-success">
            <div class="stat-number" id="correctPredictions">0</div>
            <div class="stat-label">✅ Dự Đoán Đúng</div>
        </div>
        <div class="glass stat-card stat-danger">
            <div class="stat-number" id="wrongPredictions">0</div>
            <div class="stat-label">❌ Dự Đoán Sai</div>
        </div>
    </div>
    
    <!-- Stats 2 -->
    <div class="grid-2">
        <div class="glass stat-card stat-warning">
            <div class="stat-number" id="taiRatio">0%</div>
            <div class="stat-label">📈 Tỷ Lệ Tài</div>
        </div>
        <div class="glass stat-card stat-danger">
            <div class="stat-number" id="xiuRatio">0%</div>
            <div class="stat-label">📉 Tỷ Lệ Xỉu</div>
        </div>
    </div>
    
    <!-- Pattern Chart + History -->
    <div class="grid-2">
        <div class="glass">
            <h3 style="margin-bottom:12px; font-weight:600;">📊 Biểu Đồ Mẫu Cầu (50 phiên gần nhất)</h3>
            <div class="chart-container" id="patternChart"></div>
            <div style="display:flex; justify-content:center; gap:20px; margin-top:12px;">
                <span><span style="display:inline-block; width:12px; height:12px; background:#2563eb; border-radius:3px;"></span> Tài</span>
                <span><span style="display:inline-block; width:12px; height:12px; background:#dc2626; border-radius:3px;"></span> Xỉu</span>
            </div>
        </div>
        <div class="glass">
            <h3 style="margin-bottom:12px; font-weight:600;">📋 Pattern Database (Top 5)</h3>
            <div id="patternList">
                <div style="text-align:center; color:var(--text-secondary); padding:20px;">Đang tải...</div>
            </div>
        </div>
    </div>
    
    <!-- History Table -->
    <div class="glass">
        <h3 style="margin-bottom:12px; font-weight:600;">📋 Lịch Sử Dự Đoán (50 phiên gần nhất)</h3>
        <div class="table-container" id="historyContainer">
            <table>
                <thead>
                    <tr>
                        <th>Phiên</th>
                        <th>Kết Quả</th>
                        <th>Dự Đoán</th>
                        <th>Trạng Thái</th>
                        <th>Độ Tin Cậy</th>
                        <th>Giải Thích</th>
                    </tr>
                </thead>
                <tbody id="historyBody">
                    <tr><td colspan="6" style="text-align:center; color:var(--text-secondary);">Đang tải dữ liệu...</td></tr>
                </tbody>
            </table>
        </div>
    </div>
    
</div>

<script>
    // =====================================================================
    // STATE
    // =====================================================================
    let state = {
        prediction: null,
        history: [],
        rawData: [],
        patterns: []
    };
    
    let autoRefreshInterval = null;
    const THEME_KEY = 'sunwin_theme';
    
    // =====================================================================
    // DOM REFS
    // =====================================================================
    const $ = id => document.getElementById(id);
    const predictionValue = $('predictionValue');
    const confidenceValue = $('confidenceValue');
    const confidenceFill = $('confidenceFill');
    const reasonText = $('reasonText');
    const currentPhien = $('currentPhien');
    const prevResult = $('prevResult');
    const totalPredictions = $('totalPredictions');
    const correctPredictions = $('correctPredictions');
    const wrongPredictions = $('wrongPredictions');
    const taiRatio = $('taiRatio');
    const xiuRatio = $('xiuRatio');
    const patternChart = $('patternChart');
    const patternList = $('patternList');
    const historyBody = $('historyBody');
    const autoRefreshCheck = $('autoRefresh');
    
    // =====================================================================
    // THEME
    // =====================================================================
    function toggleTheme() {
        const current = document.documentElement.getAttribute('data-theme');
        const next = current === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', next);
        localStorage.setItem(THEME_KEY, next);
    }
    
    (function loadTheme() {
        const saved = localStorage.getItem(THEME_KEY) || 'dark';
        document.documentElement.setAttribute('data-theme', saved);
    })();
    
    // =====================================================================
    // FETCH DATA
    // =====================================================================
    async function fetchPrediction() {
        try {
            // Cập nhật trạng thái loading
            predictionValue.textContent = '⏳';
            predictionValue.className = 'prediction-value';
            confidenceValue.textContent = '...';
            confidenceFill.style.width = '0%';
            reasonText.textContent = 'Đang phân tích dữ liệu...';
            
            // Cập nhật đường dẫn fetch sang /sun/vilong
            const response = await fetch('/sun/vilong');
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            state.prediction = data;
            
            // Cập nhật UI
            updatePredictionUI(data);
            
            // Lấy lịch sử
            await fetchHistory();
            await fetchPatterns();
            await updateStats();
            
        } catch (error) {
            console.error('Fetch error:', error);
            predictionValue.textContent = '⚠️';
            reasonText.textContent = 'Lỗi: ' + error.message;
        }
    }
    
    async function fetchHistory() {
        try {
            const response = await fetch('/history?limit=50');
            const data = await response.json();
            state.history = data;
            renderHistory(data);
        } catch (error) {
            console.error('History error:', error);
        }
    }
    
    async function fetchPatterns() {
        try {
            // Lấy từ dữ liệu raw
            const response = await fetch('/api/tx');
            const data = await response.json();
            if (Array.isArray(data)) {
                state.rawData = data;
                // Phân tích pattern đơn giản
                const patterns = analyzePatterns(data);
                state.patterns = patterns;
                renderPatterns(patterns);
                renderChart(data);
            }
        } catch (error) {
            console.error('Pattern error:', error);
        }
    }
    
    // =====================================================================
    // UPDATE UI
    // =====================================================================
    function updatePredictionUI(data) {
        const duDoan = data.du_doan || 'Chưa xác định';
        const doTinCay = data.do_tin_cay || 0;
        const giaiThich = data.giai_thich || 'Không có giải thích';
        const phienHienTai = data.phien_hien_tai || '---';
        const ketQuaTruoc = data.ket_qua || '---';
        
        // Prediction value
        predictionValue.textContent = duDoan;
        predictionValue.className = 'prediction-value';
        if (duDoan === 'Tài') predictionValue.classList.add('tai');
        else if (duDoan === 'Xỉu') predictionValue.classList.add('xiu');
        
        // Confidence
        confidenceValue.textContent = doTinCay + '%';
        confidenceFill.style.width = doTinCay + '%';
        
        // Reason
        reasonText.textContent = '📝 ' + giaiThich;
        
        // Meta
        currentPhien.textContent = phienHienTai;
        prevResult.textContent = ketQuaTruoc;
    }
    
    function renderHistory(history) {
        if (!history || history.length === 0) {
            historyBody.innerHTML = '<tr><td colspan="6" style="text-align:center; color:var(--text-secondary);">Chưa có dữ liệu</td></tr>';
            return;
        }
        
        let html = '';
        const reversed = history.slice().reverse();
        for (const record of reversed) {
            const ketQua = record.ket_qua || '---';
            const duDoan = record.du_doan || '---';
            let status = '⏳ Chờ';
            let statusClass = 'status-pending';
            
            if (ketQua !== '---' && duDoan !== '---') {
                const isCorrect = ketQua === duDoan;
                status = isCorrect ? '✅ Đúng' : '❌ Sai';
                statusClass = isCorrect ? 'status-correct' : 'status-wrong';
            }
            
            html += \`<tr>
                <td>\${record.phien || '---'}</td>
                <td>\${ketQua}</td>
                <td>\${duDoan}</td>
                <td class="\${statusClass}">\${status}</td>
                <td>\${record.do_tin_cay || 0}%</td>
                <td style="font-size:12px; max-width:200px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">\${record.giai_thich || '---'}</td>
            </tr>\`;
        }
        
        historyBody.innerHTML = html;
    }
    
    function renderPatterns(patterns) {
        if (!patterns || patterns.length === 0) {
            patternList.innerHTML = '<div style="text-align:center; color:var(--text-secondary); padding:20px;">Chưa có dữ liệu pattern</div>';
            return;
        }
        
        let html = '';
        for (const p of patterns.slice(0, 5)) {
            const nextClass = p.next === 'Tài' ? 'tai' : 'xiu';
            html += \`<div class="pattern-item">
                <span class="pattern-text">\${p.pattern}</span>
                <span>
                    <span class="pattern-next \${nextClass}">→ \${p.next}</span>
                    <span class="pattern-accuracy"> (\${p.accuracy})</span>
                </span>
            </div>\`;
        }
        
        patternList.innerHTML = html;
    }
    
    function renderChart(data) {
        if (!data || data.length === 0) {
            patternChart.innerHTML = '<div style="text-align:center; color:var(--text-secondary); padding:20px;">Chưa có dữ liệu</div>';
            return;
        }
        
        const recent = data.slice(0, 50);
        const maxHeight = 150;
        
        let html = '';
        for (const item of recent) {
            const result = item.ket_qua || item.KetQua || '';
            const isTai = result === 'Tài';
            const height = isTai ? maxHeight : maxHeight * 0.7;
            const cls = isTai ? 'tai' : 'xiu';
            const label = isTai ? 'T' : 'X';
            
            html += \`<div class="chart-bar \${cls}" style="height:\${height}px;">
                <span class="chart-bar-label">\${label}</span>
            </div>\`;
        }
        
        patternChart.innerHTML = html;
    }
    
    function updateStats() {
        const history = state.history || [];
        const total = history.length;
        
        let correct = 0;
        let wrong = 0;
        let taiCount = 0;
        let xiuCount = 0;
        
        for (const record of history) {
            const ketQua = record.ket_qua || '';
            const duDoan = record.du_doan || '';
            
            if (ketQua && duDoan) {
                if (ketQua === duDoan) correct++;
                else wrong++;
            }
            
            if (ketQua === 'Tài') taiCount++;
            else if (ketQua === 'Xỉu') xiuCount++;
        }
        
        totalPredictions.textContent = total;
        correctPredictions.textContent = correct;
        wrongPredictions.textContent = wrong;
        
        const totalResults = taiCount + xiuCount;
        taiRatio.textContent = totalResults > 0 ? Math.round((taiCount / totalResults) * 100) + '%' : '0%';
        xiuRatio.textContent = totalResults > 0 ? Math.round((xiuCount / totalResults) * 100) + '%' : '0%';
    }
    
    // =====================================================================
    // ANALYZE PATTERNS
    // =====================================================================
    function analyzePatterns(data) {
        const patterns = [];
        const results = data.map(item => item.ket_qua || item.KetQua || '').filter(r => r);
        
        if (results.length < 3) return patterns;
        
        // Đếm pattern 3-4 ký tự
        for (let len = 3; len <= 4; len++) {
            const patternMap = new Map();
            
            for (let i = 0; i <= results.length - len - 1; i++) {
                const pattern = results.slice(i, i + len).join('');
                const next = results[i + len];
                const key = pattern + '|' + next;
                
                if (!patternMap.has(key)) {
                    patternMap.set(key, { pattern, next, count: 0 });
                }
                patternMap.get(key).count++;
            }
            
            for (const [key, value] of patternMap) {
                if (value.count >= 2) {
                    const total = results.length - len;
                    const accuracy = Math.round((value.count / total) * 100);
                    patterns.push({
                        pattern: value.pattern,
                        next: value.next,
                        count: value.count,
                        accuracy: accuracy + '%'
                    });
                }
            }
        }
        
        patterns.sort((a, b) => b.count - a.count);
        return patterns;
    }
    
    // =====================================================================
    // EXPORT
    // =====================================================================
    function exportJSON() {
        window.open('/export/json', '_blank');
    }
    
    function exportCSV() {
        window.open('/export/csv', '_blank');
    }
    
    // =====================================================================
    // AUTO REFRESH
    // =====================================================================
    function toggleAutoRefresh() {
        if (autoRefreshCheck.checked) {
            startAutoRefresh();
        } else {
            stopAutoRefresh();
        }
    }
    
    function startAutoRefresh() {
        if (autoRefreshInterval) clearInterval(autoRefreshInterval);
        autoRefreshInterval = setInterval(fetchPrediction, 30000);
    }
    
    function stopAutoRefresh() {
        if (autoRefreshInterval) {
            clearInterval(autoRefreshInterval);
            autoRefreshInterval = null;
        }
    }
    
    autoRefreshCheck.addEventListener('change', toggleAutoRefresh);
    
    // =====================================================================
    // INIT
    // =====================================================================
    fetchPrediction();
    if (autoRefreshCheck.checked) {
        startAutoRefresh();
    }
    
    // Refresh mỗi 30s
    setInterval(() => {
        if (autoRefreshCheck.checked) {
            fetchPrediction();
        }
    }, 30000);
    
    console.log('🎲 Sunwin Dashboard loaded');
    console.log('📊 Auto-refresh:', autoRefreshCheck.checked ? 'ON' : 'OFF');
</script>

</body>
</html>`;

safeWriteFileSync(path.join(__dirname, 'public', 'dashboard.html'), dashboardHTML);

// =====================================================================
// 13. START SERVER
// =====================================================================

app.listen(PORT, '0.0.0.0', () => {
    console.log(`🚀 Sunwin API Server running on port ${PORT}`);
    console.log(`📊 Dashboard: http://localhost:${PORT}/dashboard`);
    console.log(`📡 API: http://localhost:${PORT}/sun/vilong`);
    console.log(`📥 Export JSON: http://localhost:${PORT}/export/json`);
    console.log(`📥 Export CSV: http://localhost:${PORT}/export/csv`);
});

module.exports = app;