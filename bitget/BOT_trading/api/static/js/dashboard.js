// dashboard.js===========================================================================
// BOT_trading Dashboard JavaScript
// ===========================================================================

const COLORS = {
    purple: '#6d28d9',
    green: '#3fb950',
    healthy: '#58a6ff',
    warning: '#f0883e',   // ← AÑADIR
    danger: '#f85149',    // ← AÑADIR
    yellow: '#d29922',
    red: '#f85149',
    textPrimary: '#c9d1d9',
    textSecondary: '#8b949e',
    gridDark: '#21262d',
    borderYellow: '#facc15',
    white: '#ffffff',
    blue: '#58a6ff',
    equityPositive: '#3fb950',
    equityNegative: '#f85149',
    drawdownRed: '#f85149',
    drawdownRedAlpha: 'rgba(248, 81, 73, 0.1)'
};

const METRIC_THRESHOLDS = {
    profitFactor: { excellent: 2.0, good: 1.5, acceptable: 1.0 },
    sharpeRatio: { excellent: 2.0, good: 1.5, acceptable: 1.0 },
    rSquared: { excellent: 0.9, good: 0.7, acceptable: 0.5 }
};

const CHART_DEFAULTS = {
    fontSize: { title: 20, axis: 16 },
    gridColor: COLORS.gridDark,
    borderColor: COLORS.borderYellow,
    borderWidth: 1,
    textColor: COLORS.white,
    titleColor: COLORS.textPrimary
};

let SLIPPAGE_THRESHOLDS = {
    warning: 0.2,   // Fallback si falla el fetch
    critical: 0.3
};
async function loadQualityThresholds() {
    try {
        const res = await fetch('/api/quality/thresholds');
        const data = await res.json();
        if (data.success) {
            SLIPPAGE_THRESHOLDS.warning = data.thresholds.slippage_warning_pct;
            SLIPPAGE_THRESHOLDS.critical = data.thresholds.slippage_critical_pct;
        }
    } catch (error) {
        console.error('Error loading quality thresholds:', error);
    }
}
function getMetricColor(value, withGlow = false) {
    const thresholds = METRIC_THRESHOLDS.profitFactor;
    if (value >= thresholds.excellent) {
        return { color: COLORS.purple, shadow: withGlow ? '0 0 10px rgba(109, 40, 217, 0.8)' : 'none' };
    } else if (value >= thresholds.good) {
        return { color: COLORS.green, shadow: 'none' };
    } else if (value >= thresholds.acceptable) {
        return { color: COLORS.yellow, shadow: 'none' };
    } else {
        return { color: COLORS.red, shadow: 'none' };
    }
}

function getRSquaredColor(value) {
    if (value >= 0.9) return { color: COLORS.purple, shadow: '0 0 10px rgba(109, 40, 217, 0.8)' };
    if (value >= 0.7) return { color: COLORS.green, shadow: 'none' };
    if (value >= 0.5) return { color: COLORS.yellow, shadow: 'none' };
    return { color: COLORS.red, shadow: 'none' };
}

function getPositiveNegativeColor(value) {
    return value >= 0 ? COLORS.green : COLORS.red;
}

function applyMetricColor(element, value, type = 'profitFactor') {
    if (type === 'profitFactor' || type === 'sharpe') {
        const { color, shadow } = getMetricColor(value, true);
        element.style.color = color;
        element.style.textShadow = shadow;
    } else if (type === 'rSquared') {
        const { color, shadow } = getRSquaredColor(value);
        element.style.color = color;
        element.style.textShadow = shadow;
    } else if (type === 'positiveNegative') {
        element.style.color = getPositiveNegativeColor(value);
    }
}

function getBaseChartConfig(title, reverseY = false) {
    const config = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { display: false },
            title: { 
                display: true, 
                text: title,
                color: CHART_DEFAULTS.titleColor,
                font: { size: CHART_DEFAULTS.fontSize.title, weight: 'bold' }
            }
        },
        scales: {
            x: { 
                ticks: { color: CHART_DEFAULTS.textColor, font: { size: CHART_DEFAULTS.fontSize.axis } }, 
                grid: { 
                    color: CHART_DEFAULTS.gridColor,
                    drawBorder: true,
                    borderColor: CHART_DEFAULTS.borderColor,
                    borderWidth: CHART_DEFAULTS.borderWidth
                } 
            },
            y: { 
                ticks: { 
                    color: CHART_DEFAULTS.textColor,
                    font: { size: CHART_DEFAULTS.fontSize.axis },
                    callback: function(value) { return value.toFixed(1) + '%'; }
                }, 
                grid: { 
                    color: CHART_DEFAULTS.gridColor,
                    drawBorder: true,
                    borderColor: CHART_DEFAULTS.borderColor,
                    borderWidth: CHART_DEFAULTS.borderWidth
                }
            }
        }
    };
    if (reverseY) config.scales.y.reverse = true;
    return config;
}

function clearAnalysisDates() {
    document.getElementById('analysis-date-from').value = '';
    document.getElementById('analysis-date-to').value = '';
}

function getAnalysisDateParams() {
    const dateFrom = document.getElementById('analysis-date-from').value;
    const dateTo = document.getElementById('analysis-date-to').value;
    let params = '';
    if (dateFrom) params += '&date_from=' + dateFrom;
    if (dateTo) params += '&date_to=' + dateTo;
    return params;
}

// Date filter helper functions
function clearCurvesDates() {
    document.getElementById('curves-date-from').value = '';
    document.getElementById('curves-date-to').value = '';
}

function clearComposeDates() {
    document.getElementById('compose-date-from').value = '';
    document.getElementById('compose-date-to').value = '';
}

function getCurvesDateParams() {
    const dateFrom = document.getElementById('curves-date-from').value;
    const dateTo = document.getElementById('curves-date-to').value;
    let params = '';
    if (dateFrom) params += '&date_from=' + dateFrom;
    if (dateTo) params += '&date_to=' + dateTo;
    return params;
}

function getComposeDateParams() {
    const dateFrom = document.getElementById('compose-date-from').value;
    const dateTo = document.getElementById('compose-date-to').value;
    let params = '';
    if (dateFrom) params += '&date_from=' + dateFrom;
    if (dateTo) params += '&date_to=' + dateTo;
    return params;
}

function clearRiskDates() {
    document.getElementById('risk-date-from').value = '';
    document.getElementById('risk-date-to').value = '';
}

function getRiskDateParams() {
    const dateFrom = document.getElementById('risk-date-from').value;
    const dateTo = document.getElementById('risk-date-to').value;
    let params = '';
    if (dateFrom) params += '&date_from=' + dateFrom;
    if (dateTo) params += '&date_to=' + dateTo;
    return params;
}

async function waitForBackend() {
    const maxAttempts = 30;
    let attempts = 0;
    const splash = document.getElementById('loading-splash');
    const statusText = document.getElementById('loading-status');
    const attemptCount = document.getElementById('attempt-count');
    const errorBox = document.getElementById('loading-error');
    
    splash.classList.remove('hidden');
    splash.style.opacity = '1';
    splash.style.display = 'flex';
    errorBox.classList.remove('visible');
    
    while (attempts < maxAttempts) {
        attempts++;
        attemptCount.textContent = attempts;
        
        try {
            const response = await fetch('/api/status', { 
                method: 'GET',
                cache: 'no-cache',
                headers: { 'Accept': 'application/json', 'Cache-Control': 'no-cache' }
            });
            
            if (response.ok) {
                const data = await response.json();
                if (data.status === 'running' || data.account) {
                    statusText.innerHTML = '✅ Connected successfully!';
                    statusText.style.color = '#4ade80';
                    await new Promise(r => setTimeout(r, 500));
                    splash.style.transition = 'opacity 0.3s ease';
                    splash.style.opacity = '0';
                    await new Promise(r => setTimeout(r, 300));
                    splash.classList.add('hidden');
                    splash.style.display = 'none';
                    return true;
                }
            }
        } catch (error) {}
        await new Promise(r => setTimeout(r, 1000));
    }
    
    errorBox.classList.add('visible');
    setTimeout(() => {
        splash.style.transition = 'opacity 0.5s ease';
        splash.style.opacity = '0';
        setTimeout(() => {
            splash.classList.add('hidden');
            splash.style.display = 'none';
        }, 500);
    }, 5000);
    return false;
}

let autoScroll = true;
let isLoadingData = false;
let isLoadingLogs = false;
let currentPositionsView = 'compact';
let cachedPositions = [];
let currentTab = 'positions';
let equityChart = null;
let drawdownChart = null;
let allStrategiesList = [];
let isStoppingBot = false;
let currentRegimeAnalyticsMode = 'regime';
let riskExposureChart = null;
let MAX_GROSS_EXPOSURE = 30.0;  // Default, overwritten by backend
let MAX_NET_EXPOSURE = 20.0;     // Default, overwritten by backend
// ═══════════════════════════════════════════════════════════════════════════
// POSITION SORT FEATURE (NEW)
// ═══════════════════════════════════════════════════════════════════════════

let positionSortBy = 'tp';

function sortPositionsBy(type) {
    positionSortBy = type;
    
    // Update button states
    const buttons = document.querySelectorAll('#sort-buttons .view-btn');
    buttons.forEach(btn => btn.classList.remove('active'));
    
    // Find and activate the clicked button
    buttons.forEach(btn => {
        if (btn.textContent.trim().toUpperCase() === type.toUpperCase()) {
            btn.classList.add('active');
        }
    });
    
    renderPositions(cachedPositions);
}

// ═══════════════════════════════════════════════════════════════════════════
// END POSITION SORT FEATURE
// ═══════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════
// REGIME MATRIX FUNCTIONS (NEW)
// ═══════════════════════════════════════════════════════════════════════════

function getFamilyColor(family) {
    const colors = {
        'trending': '#58a6ff',      // AZUL (cambiado de verde)
        'ranging': '#9ca3af',        // GRIS CLARO (cambiado de gris oscuro)
        'volatile': '#f85149',       // ROJO (sin cambios)
        'general': '#4b5563'
    };
    return colors[family] || '#8b949e';
}

async function loadRegimeSizing() {
    try {
        const res = await fetch('/api/regime/current?timeframe=4H');
        const data = await res.json();
        
        if (data.success && data.all_families) {
            window.REGIME_SIZING = data.all_families;
        }
    } catch (error) {
        console.error('Error loading regime sizing:', error);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MARKET REGIME FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

let currentRegimeTimeframe = '4H';

function getRegimeBadgeStyle(family) {
    const styles = {
        'trending': { bg: '#58a6ff', color: '#ffffff', text: 'TRENDING' },      // AZUL
        'volatile': { bg: '#f85149', color: '#ffffff', text: 'VOLATILE' },      // ROJO
        'ranging': { bg: '#9ca3af', color: '#ffffff', text: 'RANGING' },        // GRIS CLARO
        'default': { bg: '#6b7280', color: '#ffffff', text: 'UNKNOWN' }
    };
    return styles[family] || styles['default'];
}

function setRegimeTimeframe(timeframe) {
    currentRegimeTimeframe = timeframe;
    
    // Update active button
    document.querySelectorAll('#tab-regime .view-selector .view-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.textContent === timeframe) {
            btn.classList.add('active');
        }
    });
    
    // Load regime data
    loadRegimeData();
    loadRegime0Data();

}

async function loadRegimeData() {
    try {
        const res = await fetch('/api/regime/current?timeframe=' + currentRegimeTimeframe);
        if (!res.ok) throw new Error('HTTP ' + res.status);
        
        const data = await res.json();
        
        if (!data.success) {
            console.error('Regime API error:', data.error);
            updateRegimeUI({
                family: 'error',
                multiplier: 1.0,
                metrics: {},
                timeframe: currentRegimeTimeframe
            });
            return;
        }
        
        updateRegimeUI(data);
        
    } catch (error) {
        console.error('Error loading regime data:', error);
        updateRegimeUI({
            family: 'error',
            multiplier: 1.0,
            metrics: {},
            timeframe: currentRegimeTimeframe
        });
    }
}

function updateRegimeUI(data) {
    const family = data.family || 'default';
    const multiplier = data.multiplier || 1.0;
    const metrics = data.metrics || {};
    const timeframe = data.timeframe || currentRegimeTimeframe;
    
    // GLOBAL CONSTANTS
    const TOTAL_BLOCKS = 72;
    
    // CRITICAL: all_families must come from backend (settings.py)
    // If backend fails, show error instead of using outdated hardcoded values
    const allFamilies = data.all_families;
    
    if (!allFamilies || Object.keys(allFamilies).length === 0) {
        console.error('Backend did not return all_families - check REGIME_FAMILY_SIZING in settings.py');
        const regimeContainer = document.getElementById('tab-regime');
        if (regimeContainer) {
            regimeContainer.innerHTML = '<div style="color: #f85149; padding: 40px; text-align: center;">Error: Market regime configuration not available. Check backend logs.</div>';
        }
        return;
    }
    const allThresholds = data.all_thresholds || {};
    
    // ✅ NUEVO: Renderizar las 3 cards de BTC trend
    const btcPrice = data.btc_price;
    const btcMa50 = data.btc_ma50;
    const btcTrend = data.btc_trend;
    
    // Update BTC trend cards
    const btcCardShort = document.getElementById('btc-card-short');
    const btcCardInfo = document.getElementById('btc-card-info');
    const btcCardLong = document.getElementById('btc-card-long');
    
    if (btcCardInfo) {
        // Update center card with BTC info
        const priceEl = document.getElementById('btc-trend-price');
        const ma50El = document.getElementById('btc-trend-ma50');
        const statusEl = document.getElementById('btc-trend-status');
        
        if (priceEl && btcPrice !== undefined && btcPrice !== null) {
            priceEl.textContent = '$' + btcPrice.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
        }
        
        if (ma50El && btcMa50 !== undefined && btcMa50 !== null) {
            ma50El.textContent = '$' + btcMa50.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
        }
        
        if (statusEl && btcTrend) {
            if (btcTrend === 'uptrend') {
                statusEl.textContent = '⬆️ UPTREND';
                statusEl.style.color = '#3fb950'; // Verde
            } else if (btcTrend === 'downtrend') {
                statusEl.textContent = '⬇️ DOWNTREND';
                statusEl.style.color = '#f85149'; // Rojo
            } else {
                statusEl.textContent = '-';
                statusEl.style.color = '#c9d1d9';
            }
        }
    }
    
    // Highlight active card based on btcTrend
    if (btcCardShort && btcCardLong && btcTrend) {
        if (btcTrend === 'downtrend') {
            // SHORT card activa
            btcCardShort.style.border = '2px solid #f85149';
            btcCardShort.style.background = 'rgba(248, 81, 73, 0.15)';
            btcCardShort.style.opacity = '1';
            
            // LONG card inactiva
            btcCardLong.style.border = '2px solid #21262d';
            btcCardLong.style.background = '#1c2128';
            btcCardLong.style.opacity = '0.5';
        } else if (btcTrend === 'uptrend') {
            // LONG card activa
            btcCardLong.style.border = '2px solid #3fb950';
            btcCardLong.style.background = 'rgba(63, 185, 80, 0.15)';
            btcCardLong.style.opacity = '1';
            
            // SHORT card inactiva
            btcCardShort.style.border = '2px solid #21262d';
            btcCardShort.style.background = '#1c2128';
            btcCardShort.style.opacity = '0.5';
        } else {
            // Ambas inactivas
            btcCardShort.style.border = '2px solid #21262d';
            btcCardShort.style.background = '#1c2128';
            btcCardShort.style.opacity = '0.5';
            
            btcCardLong.style.border = '2px solid #21262d';
            btcCardLong.style.background = '#1c2128';
            btcCardLong.style.opacity = '0.5';
        }
    }
    
    // Get badge style
    const badgeStyle = getRegimeBadgeStyle(family);
    
    // Update header stat card
    const regimeText = document.getElementById('regime-text');
    const regimeDirection = document.getElementById('regime-direction');
    const regimeTimeframeEl = document.getElementById('regime-timeframe');
    
    if (regimeText) {
        regimeText.textContent = badgeStyle.text;
        regimeText.style.color = badgeStyle.bg;
    }
    
    if (regimeDirection && btcTrend) {
        let dirSymbol = '';
        let dirText = '';
        let dirColor = '#8b949e';
        
        if (btcTrend === 'uptrend') {
            dirSymbol = '';
            dirText = 'UP';
            dirColor = '#3fb950';  // Green
        } else if (btcTrend === 'downtrend' || btcTrend === 'dwtrend') {
            dirSymbol = '';
            dirText = 'DW';
            dirColor = '#f85149';  // Red
        } else {
            dirSymbol = '•';
            dirText = '--';
            dirColor = '#8b949e';  // Gray
        }
        
        regimeDirection.textContent = dirSymbol + dirText;
        regimeDirection.style.color = dirColor;
    }
    
    if (regimeTimeframeEl) {
        regimeTimeframeEl.textContent = timeframe;
    }
    
    // Helper function to format rules
    function formatRules(familyName) {
        const rules = allThresholds[familyName] || {};
        
        if (Object.keys(rules).length === 0) {
            return 'Default - All other conditions';
        }
        
        const metricNames = {
            'hurst': 'Hurst',
            'efficiency_ratio': 'ER',
            'atr_pct': 'ATR%',
            'permutation_entropy': 'PE'
        };
        
        const parts = [];
        for (const [metric, [op, threshold]] of Object.entries(rules)) {
            const name = metricNames[metric] || metric;
            parts.push(name + ' ' + op + ' ' + threshold);
        }
        
        return parts.join(' AND ');
    }
    
    // Update family cards with rules and active state (NO MULTIPLIERS)
    const families = ['volatile', 'ranging', 'trending'];
    const familyColors = {
        'volatile': '#f85149',       // ROJO
        'ranging': '#9ca3af',         // GRIS CLARO
        'trending': '#58a6ff'         // AZUL
    };
    const familyBgColors = {
        'volatile': 'rgba(248, 81, 73, 0.15)',
        'ranging': 'rgba(156, 163, 175, 0.15)',
        'trending': 'rgba(88, 166, 255, 0.15)'
    };
    
    families.forEach(f => {
        const card = document.getElementById('regime-card-' + f);
        const rulesSpan = document.getElementById('regime-card-' + f + '-rules');
        
        if (card) {
            // Set rules
            if (rulesSpan) {
                rulesSpan.textContent = formatRules(f);
            }
            
            // Set active/inactive state with colored background
            if (f === family.toLowerCase()) {
                // Active card
                card.style.border = '2px solid ' + familyColors[f];
                card.style.background = familyBgColors[f]; // Soft colored background
                card.style.opacity = '1';
            } else {
                // Inactive card
                card.style.border = '2px solid #21262d';
                card.style.background = '#1c2128'; // Normal background
                card.style.opacity = '0.5';
            }
        }
    });
    
    // Helper function to create colored block bar
    function createColoredBar(value, reverse = false) {
        const totalBlocks = TOTAL_BLOCKS;
        const filledBlocks = Math.round(value * totalBlocks);
        
        // Determine single color based on value (not position)
        let fillColor;
        
        if (reverse) {
            // For ATR and PE (lower is better): green → yellow → red
            if (value < 0.33) {
                fillColor = '#3fb950'; // Green (low value is good)
            } else if (value < 0.66) {
                fillColor = '#f59e0b'; // Yellow (medium value)
            } else {
                fillColor = '#f85149'; // Red (high value is bad)
            }
        } else {
            // For Hurst and ER (higher is better): red → yellow → green
            if (value < 0.33) {
                fillColor = '#f85149'; // Red (low value is bad)
            } else if (value < 0.66) {
                fillColor = '#f59e0b'; // Yellow (medium value)
            } else {
                fillColor = '#3fb950'; // Green (high value is good)
            }
        }
        
        // Build HTML with single color for all filled blocks
        let html = '';
        
        for (let i = 0; i < totalBlocks; i++) {
            if (i < filledBlocks) {
                html += '<span style="color: ' + fillColor + ';">█</span>';
            } else {
                html += '<span style="color: #2d333b;">█</span>';
            }
        }
        
        return html;
    }
    
    // Helper to create empty bar
    function createEmptyBar() {
        const totalBlocks = TOTAL_BLOCKS;
        let html = '';
        for (let i = 0; i < totalBlocks; i++) {
            html += '<span style="color: #2d333b;">█</span>';
        }
        return html;
    }
    
    // Update metrics with colored block bars
    const hurst = metrics.hurst;
    const er = metrics.efficiency_ratio;
    const atr = metrics.atr_pct;
    const pe = metrics.permutation_entropy;
    
    // Hurst Exponent (0-1 range, higher is better)
    if (hurst !== undefined && hurst !== null && !isNaN(hurst)) {
        const hurstElement = document.getElementById('regime-metric-hurst');
        hurstElement.textContent = hurst.toFixed(3);
        
        // Set color to match bar color
        let hurstColor;
        if (hurst < 0.33) {
            hurstColor = '#f85149'; // Red (low value is bad)
        } else if (hurst < 0.66) {
            hurstColor = '#f59e0b'; // Yellow (medium value)
        } else {
            hurstColor = '#3fb950'; // Green (high value is good)
        }
        hurstElement.style.color = hurstColor;
        
        const hurstBar = document.getElementById('regime-bar-hurst');
        if (hurstBar) {
            hurstBar.innerHTML = createColoredBar(hurst, false);
            hurstBar.style.width = '100%';
        }
    } else {
        document.getElementById('regime-metric-hurst').textContent = '-';
        document.getElementById('regime-metric-hurst').style.color = '#58a6ff';
        const hurstBar = document.getElementById('regime-bar-hurst');
        if (hurstBar) {
            hurstBar.innerHTML = createEmptyBar();
            hurstBar.style.width = '100%';
        }
    }
    
    // Efficiency Ratio (0-1 range, higher is better)
    if (er !== undefined && er !== null && !isNaN(er)) {
        const erElement = document.getElementById('regime-metric-er');
        erElement.textContent = er.toFixed(3);
        
        // Set color to match bar color
        let erColor;
        if (er < 0.33) {
            erColor = '#f85149'; // Red (low value is bad)
        } else if (er < 0.66) {
            erColor = '#f59e0b'; // Yellow (medium value)
        } else {
            erColor = '#3fb950'; // Green (high value is good)
        }
        erElement.style.color = erColor;
        
        const erBar = document.getElementById('regime-bar-er');
        if (erBar) {
            erBar.innerHTML = createColoredBar(er, false);
            erBar.style.width = '100%';
        }
    } else {
        document.getElementById('regime-metric-er').textContent = '-';
        document.getElementById('regime-metric-er').style.color = '#22d3ee';
        const erBar = document.getElementById('regime-bar-er');
        if (erBar) {
            erBar.innerHTML = createEmptyBar();
            erBar.style.width = '100%';
        }
    }
    
    // ATR Percentage (scale 0-4% to 0-1, lower is better)
    if (atr !== undefined && atr !== null && !isNaN(atr)) {
        const atrElement = document.getElementById('regime-metric-atr');
        atrElement.textContent = atr.toFixed(2) + '%';
        
        // Color for text based on absolute ATR value
        let atrColor;
        if (atr > 2.0) {
            atrColor = '#f85149'; // Red if > 2%
        } else if (atr > 1.0) {
            atrColor = '#f59e0b'; // Yellow if 1-2%
        } else {
            atrColor = '#3fb950'; // Green if < 1%
        }
        atrElement.style.color = atrColor;
        
        // Bar uses SAME color logic (not reverse gradient)
        const atrScaled = Math.min(atr / 2.5, 1.0);
        const totalBlocks = TOTAL_BLOCKS;
        const filledBlocks = Math.round(atrScaled * totalBlocks);
        
        // Determine bar color based on actual ATR value
        let barFillColor;
        if (atr > 2.0) {
            barFillColor = '#f85149'; // Red
        } else if (atr > 1.0) {
            barFillColor = '#f59e0b'; // Yellow
        } else {
            barFillColor = '#3fb950'; // Green
        }
        
        let barHtml = '';
        for (let i = 0; i < totalBlocks; i++) {
            if (i < filledBlocks) {
                barHtml += '<span style="color: ' + barFillColor + ';">█</span>';
            } else {
                barHtml += '<span style="color: #2d333b;">█</span>';
            }
        }
        
        const atrBar = document.getElementById('regime-bar-atr');
        if (atrBar) {
            atrBar.innerHTML = barHtml;
            atrBar.style.width = '100%';
        }
    } else {
        document.getElementById('regime-metric-atr').textContent = '-';
        document.getElementById('regime-metric-atr').style.color = '#f59e0b';
        const atrBar = document.getElementById('regime-bar-atr');
        if (atrBar) {
            atrBar.innerHTML = createEmptyBar();
            atrBar.style.width = '100%';
        }
    }
    
    // Permutation Entropy (0-1 range, lower is better)
    if (pe !== undefined && pe !== null && !isNaN(pe)) {
        const peElement = document.getElementById('regime-metric-pe');
        peElement.textContent = pe.toFixed(3);
        
        // Set color to match bar color (reverse - lower is better)
        let peColor;
        if (pe < 0.33) {
            peColor = '#3fb950'; // Green (low value is good)
        } else if (pe < 0.66) {
            peColor = '#f59e0b'; // Yellow (medium value)
        } else {
            peColor = '#f85149'; // Red (high value is bad)
        }
        peElement.style.color = peColor;
        
        const peBar = document.getElementById('regime-bar-pe');
        if (peBar) {
            peBar.innerHTML = createColoredBar(pe, true);
            peBar.style.width = '100%';
        }
    } else {
        document.getElementById('regime-metric-pe').textContent = '-';
        document.getElementById('regime-metric-pe').style.color = '#a78bfa';
        const peBar = document.getElementById('regime-bar-pe');
        if (peBar) {
            peBar.innerHTML = createEmptyBar();
            peBar.style.width = '100%';
        }
    }
}

// REGIME 0 (BTC 1D FILTER) FUNCTIONS
async function loadRegime0Data() {
    try {
        const res = await fetch('/api/regime0/current');
        if (!res.ok) throw new Error('HTTP ' + res.status);
        
        const data = await res.json();
        
        if (!data.success) {
            console.error('Regime 0 API error:', data.error);
            return;
        }
        
        updateRegime0UI(data);
        
    } catch (error) {
        console.error('Error loading REGIME 0 data:', error);
    }
}

function updateRegime0UI(data) {
    const btcClose = data.btc_close;
    const ma5 = data.ma5;
    const longData = data.long;
    const shortData = data.short;
    
    const longAllowed = longData.allowed;
    const shortAllowed = shortData.allowed;
    const noneActive = !longAllowed && !shortAllowed;
    
    // Update SHORT card (LEFT) - No numbers
    const shortCard = document.getElementById('regime0-card-short');
    const shortStatus = document.getElementById('regime0-short-status');
    
    if (shortAllowed) {
        shortCard.style.border = '2px solid #f85149';
        shortCard.style.background = 'rgba(248, 81, 73, 0.1)';
        shortStatus.textContent = 'ALLOW';
        shortStatus.style.color = '#f85149';
    } else {
        shortCard.style.border = '2px solid #21262d';
        shortCard.style.background = '#1c2128';
        shortStatus.textContent = 'BLOCK';
        shortStatus.style.color = '#6b7280';
    }
    
    // Update INACTIVE card (CENTER) - 3 values only
    const noneCard = document.getElementById('regime0-card-none');
    const noneShortTh = document.getElementById('regime0-none-short-th');
    const noneBtc = document.getElementById('regime0-none-btc');
    const noneLongTh = document.getElementById('regime0-none-long-th');
    
    if (noneActive) {
        noneCard.style.border = '2px solid #d29922';
        noneCard.style.background = 'rgba(210, 153, 34, 0.1)';
    } else {
        noneCard.style.border = '2px solid #21262d';
        noneCard.style.background = '#1c2128';
    }
    
    // Update 3 values in INACTIVE card
    if (noneShortTh && shortData.threshold) {
        noneShortTh.textContent = '$' + shortData.threshold.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
    }
    if (noneBtc && btcClose) {
        noneBtc.textContent = '$' + btcClose.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
    }
    if (noneLongTh && longData.threshold) {
        noneLongTh.textContent = '$' + longData.threshold.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
    }
    
    // Update LONG card (RIGHT) - No numbers
    const longCard = document.getElementById('regime0-card-long');
    const longStatus = document.getElementById('regime0-long-status');
    
    if (longAllowed) {
        longCard.style.border = '2px solid #3fb950';
        longCard.style.background = 'rgba(63, 185, 80, 0.1)';
        longStatus.textContent = 'ALLOW';
        longStatus.style.color = '#3fb950';
    } else {
        longCard.style.border = '2px solid #21262d';
        longCard.style.background = '#1c2128';
        longStatus.textContent = 'BLOCK';
        longStatus.style.color = '#6b7280';
    }
}
// ═══════════════════════════════════════════════════════════════════════════
// END MARKET REGIME FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

async function stopBot() {
    if (isStoppingBot) return;
    if (!confirm('WARNING: STOP TRADING BOT?\n\nThis will terminate the bot process.\n\nAre you sure?')) return;
    isStoppingBot = true;
    
    const overlay = document.getElementById('stop-overlay');
    const statusDiv = document.getElementById('stop-status');
    statusDiv.innerHTML = '';
    overlay.classList.add('active');
    
    addStopLog('Initializing stop sequence...');
    addStopLog('Stop signal requested');
    
    try {
        const stopResponse = await fetch('/api/bot/stop', { method: 'POST' });
        if (!stopResponse.ok) throw new Error('Failed to send stop signal');
        
        const stopData = await stopResponse.json();
        addStopLog('Stop signal sent (SIGTERM) to PID ' + stopData.pid, 'success');
        
        let attempts = 0;
        const maxAttempts = 30;
        let botStoppedConfirmed = false;
        
        const verifyInterval = setInterval(async () => {
            attempts++;
            addStopLog('Verifying shutdown... (' + attempts + '/' + maxAttempts + ')');
            
            try {
                const verifyResponse = await fetch('/api/bot/verify-stopped', {
                    method: 'GET',
                    cache: 'no-cache'
                });
                
                if (verifyResponse.ok) {
                    const status = await verifyResponse.json();
                    if (!status.running) {
                        clearInterval(verifyInterval);
                        botStoppedConfirmed = true;
                        addStopLog('Process verified as terminated', 'success');
                        addStopLog('✅ BOT STOPPED SUCCESSFULLY', 'success');
                        document.getElementById('stop-title').textContent = '✅ BOT STOPPED';
                        document.getElementById('stop-title').style.color = COLORS.green;
                        return;
                    }
                }
            } catch (error) {
                clearInterval(verifyInterval);
                botStoppedConfirmed = true;
                addStopLog('Backend stopped responding', 'success');
                addStopLog('Flask server terminated (expected)', 'success');
                addStopLog('✅ BOT STOPPED SUCCESSFULLY', 'success');
                document.getElementById('stop-title').textContent = '✅ BOT STOPPED';
                document.getElementById('stop-title').style.color = COLORS.green;
                return;
            }
            
            if (attempts >= maxAttempts && !botStoppedConfirmed) {
                clearInterval(verifyInterval);
                addStopLog('⚠️ VERIFICATION TIMEOUT', 'warning');
                addStopLog('Bot may still be running. Check manually.', 'warning');
                document.getElementById('stop-title').textContent = '⚠️ TIMEOUT';
                document.getElementById('stop-title').style.color = COLORS.yellow;
            }
        }, 1000);
        
    } catch (error) {
        if (error.message.includes('fetch') || error.name === 'TypeError') {
            addStopLog('⚠️ Backend not responding', 'warning');
            addStopLog('Bot may have already stopped', 'warning');
        } else {
            addStopLog('❌ ERROR: ' + error.message, 'error');
        }
    }
}

function addStopLog(message, className = '') {
    const statusDiv = document.getElementById('stop-status');
    const line = document.createElement('div');
    line.className = 'stop-status-line' + (className ? ' ' + className : '');
    line.textContent = message;
    statusDiv.appendChild(line);
    statusDiv.scrollTop = statusDiv.scrollHeight;
}

function switchTab(tabName) {
    currentTab = tabName;
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
    document.getElementById('tab-' + tabName).classList.add('active');
    
    if (tabName === 'analysis') loadStrategyAnalysis();
    if (tabName === 'config') loadBotConfig();
    if (tabName === 'equity') loadEquityTab();
    if (tabName === 'regime') {
    loadRegimeData();
    loadRegime0Data();
}
    if (tabName === 'risk') loadRiskTab();
    if (tabName === 'quality') loadQualityTab();
}

function switchEquitySubTab(subTabName) {
    document.querySelectorAll('#tab-equity .tabs-container .tab-btn').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');
    document.querySelectorAll('#tab-equity .tab-content').forEach(content => content.classList.remove('active'));
    document.getElementById('equity-subtab-' + subTabName).classList.add('active');
    
    if (subTabName === 'symbols') loadSymbolsAnalysis();
    if (subTabName === 'weekday') loadWeekDayAnalysis();
    if (subTabName === 'monthly') initMonthlyTab();
    if (subTabName === 'correlation') initCorrelationTab();
    if (subTabName === 'regime') loadRegimeAnalytics();
    if (subTabName === 'compose') {
        const container = document.getElementById('compose-container');
        const currentContent = container.innerHTML.trim();
        
        if (currentContent === '' || currentContent.includes('Select a metric')) {
            container.innerHTML = '<div style="text-align: center; color: #8b949e; padding: 40px;">Select a metric and click "Show TOP 10"</div>';
            document.getElementById('compose-plot-btn').style.display = 'none';
        }
    }
}

function getLogClass(line) {
    if (line.includes('TP for')) return 'tp-hit';
    if (line.includes('SL for')) return 'sl-hit';
    // Check que NO sea SHORTS/LONGS (evitar false positives)
    if (!line.includes('SHORTS') && !line.includes('LONGS')) {
        if (line.includes('SHORT') || line.includes('LONG')) return 'info';
    }
    if (line.includes('Error')) return 'error';
    if (line.includes('WAR')) return 'warning';
    return 'default';
}


async function loadLogs() {
    if (isLoadingLogs) return;
    isLoadingLogs = true;
    try {
        const res = await fetch('/api/logs/stream');
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const data = await res.json();
        
        if (data.logs && data.logs.length > 0) {
            const logsContent = document.getElementById('logs-content');
            if (logsContent.children.length === 1 && logsContent.children[0].textContent.includes('Waiting')) {
                logsContent.innerHTML = '';
            }
            
            const fragment = document.createDocumentFragment();
            data.logs.forEach(log => {
                const logLine = document.createElement('div');
                logLine.className = 'log-line ' + getLogClass(log);
                logLine.textContent = log;
                fragment.appendChild(logLine);
            });
            logsContent.appendChild(fragment);
            
            while (logsContent.children.length > 500) {
                logsContent.removeChild(logsContent.firstChild);
            }
            
            if (autoScroll) {
                requestAnimationFrame(() => {
                    logsContent.scrollTop = logsContent.scrollHeight;
                });
            }
        }
    } catch (error) {
        console.error('Error loading logs:', error);
    } finally {
        isLoadingLogs = false;
    }
}

function setPositionsView(view) {
    currentPositionsView = view;
    document.querySelectorAll('.view-btn').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');
    
    if (view !== 'symbols') {
        const headerSpan = document.querySelector('#tab-positions .content-section h2 span');
        if (headerSpan) {
            headerSpan.textContent = 'Active Positions';
        }
    }
    
    renderPositions(cachedPositions);
    localStorage.setItem('positionsView', view);
}

function renderPositions(positions) {
    cachedPositions = positions;
    const container = document.getElementById('positions-container');
    
    if (!positions || positions.length === 0) {
        container.innerHTML = '<div style="text-align: center; color: #8b949e; padding: 40px;">No active positions</div>';
        return;
    }
    
    if (currentPositionsView === 'compact') {
        renderCompactView(container, positions);
    } else if (currentPositionsView === 'detailed') {
        renderDetailedView(container, positions);
    } else if (currentPositionsView === 'symbols') {
        renderSymbolsView(container, positions);
    }
}

function renderCompactView(container, positions) {
    const groupedByStrategy = {};
    positions.forEach(pos => {
        if (!groupedByStrategy[pos.strategy]) {
            groupedByStrategy[pos.strategy] = {
                positions: [],
                totalPnl: 0,
                direction: pos.direction,
                opened_at: pos.opened_at,
                candles: pos.candles,
                max_candles: pos.max_candles
            };
        }
        groupedByStrategy[pos.strategy].positions.push(pos);
        groupedByStrategy[pos.strategy].totalPnl += (pos.current_pnl || 0);
    });
    
    let allEntries = [];
    
    if (allStrategiesList && allStrategiesList.length > 0) {
        const allActiveDeprecating = allStrategiesList.filter(s => 
            s.status === 'ACTIVE' || s.status === 'DEPRECATING'
        );
        
        allActiveDeprecating.forEach(strat => {
            if (groupedByStrategy[strat.id]) {
                allEntries.push([strat.id, groupedByStrategy[strat.id]]);
            } else {
                allEntries.push([strat.id, {
                    positions: [],
                    totalPnl: 0,
                    direction: strat.direction,
                    opened_at: null,
                    candles: 0,
                    max_candles: 50,
                    isEmpty: true
                }]);
            }
        });
    } else {
        allEntries = Object.entries(groupedByStrategy);
    }
    
    const sortedEntries = allEntries.sort((a, b) => a[0].localeCompare(b[0]));
    
    if (sortedEntries.length === 0) {
        container.innerHTML = '<div style="text-align: center; color: #8b949e; padding: 40px;">No active positions</div>';
        return;
    }
    
    const html = '<table><thead><tr><th>#</th><th>Strategy</th><th>Side</th><th>Opened</th><th style="text-align: right;">Candles</th><th style="text-align: center;">#pos</th><th>PnL</th></tr></thead><tbody>' +
        sortedEntries.map(([strategyId, data], index) => {
            const pnl = data.totalPnl;
            const pnlClass = pnl >= 0 ? 'direction-long' : 'direction-short';
            const num = String(index + 1).padStart(2, '0');
            
            let openedDateStr = '-';
            if (data.opened_at) {
                try {
                    const date = new Date(data.opened_at);
                    openedDateStr = date.toISOString().split('T')[0];
                } catch {
                    openedDateStr = String(data.opened_at).substring(0, 10);
                }
            }
            
            if (data.isEmpty) {
                return '<tr><td style="color: #8b949e; font-weight: 600;">' + num + '</td><td>' + strategyId + '</td><td class="direction-' + data.direction.toLowerCase() + '">' + data.direction.toUpperCase() + '</td><td>-</td><td style="text-align: right;">-</td><td style="text-align: center; color: #f85149; font-weight: 600;">0</td><td>-</td></tr>';
            }
            
            return '<tr><td style="color: #8b949e; font-weight: 600;">' + num + '</td><td>' + strategyId + '</td><td class="direction-' + data.direction.toLowerCase() + '">' + data.direction.toUpperCase() + '</td><td>' + openedDateStr + '</td><td style="text-align: right;">' + (data.candles || 0) + '/' + (data.max_candles || 50) + '</td><td style="text-align: center; color: #58a6ff; font-weight: 600;">' + data.positions.length + '</td><td class="' + pnlClass + '">' + (pnl >= 0 ? '+' : '') + '$' + pnl.toFixed(2) + '</td></tr>';
        }).join('') +
        '</tbody></table>';
    container.innerHTML = html;
}

function renderDetailedView(container, positions) {
    // MODIFIED: Use positionSortBy to determine sort field
    const positionsWithDelta = positions.map(pos => {
        const currentPrice = parseFloat(pos.current_price || pos.entry_price);
        const tp = parseFloat(pos.tp);
        const sl = parseFloat(pos.sl);
        
        let deltaTp, deltaSl;
        if (pos.direction.toLowerCase() === 'long') {
            deltaTp = ((tp - currentPrice) / currentPrice * 100);
            deltaSl = ((currentPrice - sl) / currentPrice * 100);
        } else {
            deltaTp = ((currentPrice - tp) / currentPrice * 100);
            deltaSl = ((sl - currentPrice) / currentPrice * 100);
        }
        
        return { ...pos, _deltaTp: deltaTp, _deltaSl: deltaSl };
    });
    
    // MODIFIED: Sort based on positionSortBy variable
    const sortedPositions = positionsWithDelta.sort((a, b) => {
        if (positionSortBy === 'tp') {
            return a._deltaTp - b._deltaTp;
        } else {
            return a._deltaSl - b._deltaSl;
        }
    });
    
    // Determine active button class
    const tpBtnClass = positionSortBy === 'tp' ? 'view-btn active' : 'view-btn';
    const slBtnClass = positionSortBy === 'sl' ? 'view-btn active' : 'view-btn';
    
    const html = '<table><thead><tr>' +
        '<th>Strategy</th>' +
        '<th>Symbol</th>' +
        '<th>Side</th>' +
        '<th>Entry</th>' +
        '<th>Current</th>' +
        '<th>Amount</th>' +
        '<th>TP (Δ%) <button class="' + tpBtnClass + '" onclick="sortPositionsBy(\'tp\')" style="margin-left: 8px; font-size: 11px; padding: 2px 8px;">TP</button></th>' +
        '<th>SL (Δ%) <button class="' + slBtnClass + '" onclick="sortPositionsBy(\'sl\')" style="margin-left: 8px; font-size: 11px; padding: 2px 8px;">SL</button></th>' +
        '<th>PnL</th>' +
        '<th style="text-align: right;">Candles</th>' +
        '</tr></thead><tbody>' +
        sortedPositions.map(pos => {
            const pnl = pos.current_pnl || 0;
            const pnlClass = pnl >= 0 ? 'direction-long' : 'direction-short';
            
            const currentPrice = pos.current_price;
            const tp = pos.tp;
            const sl = pos.sl;
            const entryPrice = pos.entry_price;
            const precision = pos.precision || 2;
            
            const deltaTp = pos.distance_to_tp_pct || 0;
            const deltaSl = pos.distance_to_sl_pct || 0;
            
            const deltaTpClass = Math.abs(deltaTp) < 1 ? 'delta-tp-close' : 'delta-tp';
            const deltaSlClass = Math.abs(deltaSl) < 1 ? 'delta-sl-close' : 'delta-sl';
            
            return '<tr>' +
                '<td>' + pos.strategy + '</td>' +
                '<td>' + pos.symbol + '</td>' +
                '<td class="direction-' + pos.direction.toLowerCase() + '">' + pos.direction.toUpperCase() + '</td>' +
                '<td>$' + entryPrice.toFixed(precision) + '</td>' +
                '<td>$' + currentPrice.toFixed(precision) + '</td>' +
                '<td>$' + parseFloat(pos.usdt_amount).toFixed(2) + '</td>' +
                '<td>$' + tp.toFixed(precision) + ' <span class="' + deltaTpClass + '">(Δ' + (deltaTp >= 0 ? '+' : '') + deltaTp.toFixed(2) + '%)</span></td>' +
                '<td>$' + sl.toFixed(precision) + ' <span class="' + deltaSlClass + '">(Δ' + (deltaSl >= 0 ? '+' : '') + deltaSl.toFixed(2) + '%)</span></td>' +
                '<td class="' + pnlClass + '">' + (pnl >= 0 ? '+' : '') + '$' + pnl.toFixed(2) + '</td>' +
                '<td style="text-align: right;">' + (pos.candles || 0) + '/' + (pos.max_candles || 50) + '</td>' +
                '</tr>';
        }).join('') +
        '</tbody></table>';
    container.innerHTML = html;
}

function renderSymbolsView(container, positions) {
    const groupedBySymbolSide = {};
    positions.forEach(pos => {
        const key = pos.symbol + '_' + pos.direction.toUpperCase();
        
        if (!groupedBySymbolSide[key]) {
            groupedBySymbolSide[key] = {
                symbol: pos.symbol,
                side: pos.direction.toUpperCase(),
                totalSize: 0,
                totalPnl: 0,
                strategies: new Set()
            };
        }
        groupedBySymbolSide[key].totalSize += parseFloat(pos.size);
        groupedBySymbolSide[key].totalPnl += (pos.current_pnl || 0);
        
        const strategyNumber = extractNumberFromId(pos.strategy);
        groupedBySymbolSide[key].strategies.add(strategyNumber);
    });
    
    const sortedKeys = Object.keys(groupedBySymbolSide).sort((a, b) => {
        const dataA = groupedBySymbolSide[a];
        const dataB = groupedBySymbolSide[b];
        
        if (dataA.symbol !== dataB.symbol) {
            return dataA.symbol.localeCompare(dataB.symbol);
        }
        if (dataA.side === 'LONG' && dataB.side === 'SHORT') return -1;
        if (dataA.side === 'SHORT' && dataB.side === 'LONG') return 1;
        return 0;
    });
    
    if (sortedKeys.length === 0) {
        container.innerHTML = '<div style="text-align: center; color: #8b949e; padding: 40px;">No active positions</div>';
        return;
    }
    
    let longCount = 0;
    let shortCount = 0;
    sortedKeys.forEach(key => {
        if (groupedBySymbolSide[key].side === 'LONG') longCount++;
        else if (groupedBySymbolSide[key].side === 'SHORT') shortCount++;
    });
    
    const headerSpan = document.querySelector('#tab-positions .content-section h2 span');
    if (headerSpan) {
        headerSpan.textContent = 'Active Positions - 🪙 ' + sortedKeys.length + ' Unique Positions (' + longCount + ' LONG / ' + shortCount + ' SHORT)';
    }
    
    const html = '<table><thead><tr>' +
        '<th>#</th>' +
        '<th>Symbol</th>' +
        '<th>Side</th>' +
        '<th>Total Size</th>' +
        '<th>Strategies</th>' +
        '<th>PnL</th>' +
        '</tr></thead><tbody>' +
        sortedKeys.map((key, index) => {
            const data = groupedBySymbolSide[key];
            const pnlClass = data.totalPnl >= 0 ? 'direction-long' : 'direction-short';
            const sideClass = data.side === 'LONG' ? 'direction-long' : 'direction-short';
            const num = String(index + 1).padStart(2, '0');
            
            const stratNumbers = Array.from(data.strategies).sort((a, b) => {
                const numA = parseInt(a) || 0;
                const numB = parseInt(b) || 0;
                return numA - numB;
            });
            const stratDisplay = '[' + stratNumbers.join(', ') + ']';
            
            return '<tr>' +
                '<td style="color: #8b949e; font-weight: 600;">' + num + '</td>' +
                '<td style="color: #58a6ff; font-weight: 600;">' + data.symbol + '</td>' +
                '<td class="' + sideClass + '">' + data.side + '</td>' +
                '<td>' + data.totalSize.toFixed(2) + '</td>' +
                '<td style="color: #c9d1d9;">' + stratDisplay + '</td>' +
                '<td class="' + pnlClass + '">' + (data.totalPnl >= 0 ? '+' : '') + '$' + data.totalPnl.toFixed(2) + '</td>' +
                '</tr>';
        }).join('') +
        '</tbody></table>';
    
    container.innerHTML = html;
}

async function loadStrategyAnalysis() {
    try {
        const dateParams = getAnalysisDateParams();
        const res = await fetch('/api/strategy-analysis?' + dateParams);
        const data = await res.json();
        const container = document.getElementById('analysis-container');
        
        if (!data || data.length === 0) {
            container.innerHTML = '<div style="text-align: center; color: #8b949e; padding: 40px;">No data</div>';
            return;
        }
        
        const sortedData = data.sort((a, b) => a.Strategy.localeCompare(b.Strategy));
        
        const html = '<table><thead><tr><th>#</th><th>Strategy</th><th>First</th><th>Trades</th><th>Win %</th><th>Profit</th><th>Profit %</th><th>Total %</th><th>TP %</th><th>SL %</th><th>TIMEOUT %</th><th>OOM %</th><th>Avg Days</th></tr></thead><tbody>' +
            sortedData.map((s, index) => {
                const num = String(index + 1).padStart(2, '0');
                const profitClass = s.Total_profit >= 0 ? 'direction-long' : 'direction-short';
                return '<tr><td style="color: #8b949e; font-weight: 600;">' + num + '</td><td>' + s.Strategy + '</td><td>' + s.date_fo + '</td><td>' + s.Trades_num + '</td><td>' + s.Trades_pct.toFixed(1) + '%</td><td class="' + profitClass + '">' + (s.Total_profit >= 0 ? '+' : '') + '$' + s.Total_profit.toFixed(2) + '</td><td class="' + profitClass + '">' + (s.Profit_pct >= 0 ? '+' : '') + s.Profit_pct.toFixed(1) + '%</td><td>' + (s.Total_pct >= 0 ? '+' : '') + s.Total_pct.toFixed(1) + '%</td><td>' + s.TP_pct.toFixed(1) + '%</td><td>' + s.SL_pct.toFixed(1) + '%</td><td>' + s.TIMEOUT_pct.toFixed(1) + '%</td><td>' + s.OOM_pct.toFixed(1) + '%</td><td>' + s.Avg_days.toFixed(2) + '</td></tr>';
            }).join('') +
            '</tbody></table>';
        container.innerHTML = html;
    } catch (error) {
        console.error('Error:', error);
    }
}

// =============================================================================
// REGIME STRATEGY MATRIX
// =============================================================================
async function renderRegimeStrategyMatrix(strategies) {
    try {
        const res = await fetch('/api/regime/strategies');
        const data = await res.json();
        
        if (!data.success) {
            console.error('Failed to load regime strategies:', data.error);
            return;
        }
        
        const tbody = document.getElementById('regime-strategy-matrix-body');
        if (!tbody) return;
        
        const strategiesData = data.strategies || [];
        
        if (strategiesData.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" style="text-align: center;">No strategies</td></tr>';
            return;
        }
        
        // Sort by ID
        const sorted = strategiesData.sort((a, b) => a.id.localeCompare(b.id));
        
        let html = '';
        sorted.forEach((strat, idx) => {
            const num = String(idx + 1).padStart(2, '0');
            
            // Color coding for multipliers
            // Color coding for multipliers
            const getMultiplierColor = (val) => {
                if (val === '-' || val === 0) return '#6b7280';  // Gray for blocked
                if (val === 1) return '#c9d1d9';  // White for 1x
                if (val > 1) return '#58a6ff';   // Blue for >1x
                return '#f85149';  // Red for <1x
            };
            
            // Color coding for direction_mode
            let dirColor;
            if (strat.direction_mode === 'long_only') {
                dirColor = '#3fb950';  // Green
            } else if (strat.direction_mode === 'short_only') {
                dirColor = '#f85149';  // Red
            } else {
                dirColor = '#8b949e';  // Gray for general
            }
            
            const trendVal = strat.regime_trending;
            const rangVal = strat.regime_ranging;
            const volVal = strat.regime_volatile;
            
            html += `<tr>
                <td style="color: #8b949e; font-weight: 600;">${num}</td>
                <td>${strat.id}</td>
                <td style="color: ${getMultiplierColor(trendVal)}; font-weight: 700;">${trendVal !== '-' ? trendVal + 'x' : '-'}</td>
                <td style="color: ${getMultiplierColor(rangVal)}; font-weight: 700;">${rangVal !== '-' ? rangVal + 'x' : '-'}</td>
                <td style="color: ${getMultiplierColor(volVal)}; font-weight: 700;">${volVal !== '-' ? volVal + 'x' : '-'}</td>
                <td style="color: ${dirColor}; font-weight: 600; text-transform: uppercase;">${strat.direction_mode}</td>
            </tr>`;
        });
        
        tbody.innerHTML = html;
        
    } catch (error) {
        console.error('Error rendering regime strategy matrix:', error);
    }
}
// =============================================================================
// END REGIME STRATEGY MATRIX
// =============================================================================

async function loadBotConfig() {
    try {
        const res = await fetch('/api/bot-config');
        const data = await res.json();
        
        allStrategiesList = data.strategies || [];
        
        document.getElementById('config-account').textContent = data.account || '-';
        document.getElementById('config-capital').textContent = '$' + (data.initial_capital || 0).toLocaleString();
        document.getElementById('config-total-strat').textContent = data.stats.total || 0;
        document.getElementById('config-active-strat').textContent = data.stats.active || 0;
        document.getElementById('config-deprecating-strat').textContent = data.stats.deprecating || 0;
        document.getElementById('config-not-implemented-strat').textContent = data.stats.not_implemented || 0;
        
        const wsPublic = document.getElementById('ws-public');
        const wsPrivate = document.getElementById('ws-private');
        const wsAuth = document.getElementById('ws-auth');
        
        if (data.websocket_status.public_connected) {
            wsPublic.innerHTML = '<span class="ws-dot connected"></span><span>Connected</span>';
        } else {
            wsPublic.innerHTML = '<span class="ws-dot disconnected"></span><span>Disconnected</span>';
        }
        
        if (data.websocket_status.private_connected) {
            wsPrivate.innerHTML = '<span class="ws-dot connected"></span><span>Connected</span>';
        } else {
            wsPrivate.innerHTML = '<span class="ws-dot disconnected"></span><span>Disconnected</span>';
        }
        
        if (data.websocket_status.authenticated) {
            wsAuth.innerHTML = '<span class="ws-dot connected"></span><span>Authenticated</span>';
        } else {
            wsAuth.innerHTML = '<span class="ws-dot disconnected"></span><span>Not Authenticated</span>';
        }
        
        const strategiesBody = document.getElementById('strategies-body');
        if (!data.strategies || data.strategies.length === 0) {
            strategiesBody.innerHTML = '<tr><td colspan="15" style="text-align: center;">No strategies</td></tr>';
        } else {
            const sortedStrategies = data.strategies.sort((a, b) => a.id.localeCompare(b.id));
            
            // ✅ MODIFICACIÓN: Añadido 'dir_mode' a fixedKeys
            const fixedKeys = ['id', 'name', 'number', 'timeframe', 'status', 'symbols_count'];
            const commonKeys = ['tp_pct', 'sl_pct', 'order_amount', 'sell_after_ncandles'];
            const excludeKeys = new Set([...fixedKeys, ...commonKeys, 'direction', 'family_sizing', 'direction_mode', 'active', 'regime_trending', 'regime_ranging', 'regime_volatile']);
            
            const extraParamKeys = new Set();
            sortedStrategies.forEach(strat => {
                Object.keys(strat).forEach(key => {
                    if (!excludeKeys.has(key)) {
                        extraParamKeys.add(key);
                    }
                });
            });
            
            const extraParamKeysArray = Array.from(extraParamKeys).sort();
            
            let dynamicHeaders = '';
            extraParamKeysArray.forEach(key => {
                const firstWord = key.split('_')[0];
                const displayName = firstWord.charAt(0).toUpperCase() + firstWord.slice(1);
                dynamicHeaders += '<th>' + displayName + '</th>';
            });
            
            // ✅ MODIFICACIÓN: Añadida columna Dir Mode
            document.querySelector('#strategies-table thead tr').innerHTML = 
                            '<th>#</th>' +
                            '<th>ID</th>' +
                            '<th>TF</th>' +
                            '<th>Symbols</th>' +
                            '<th>TP%</th>' +
                            '<th>SL%</th>' +
                            '<th>Amount</th>' +
                            '<th>Candles</th>' +
                            dynamicHeaders +
                            '<th>Status</th>';
            
            strategiesBody.innerHTML = sortedStrategies.map((strat, index) => {
                let statusBadge = '';
                if (strat.status === 'ACTIVE') {
                    statusBadge = '<span class="badge badge-active">Active</span>';
                } else if (strat.status === 'DEPRECATING') {
                    statusBadge = '<span class="badge badge-deprecating">Deprecating</span>';
                } else {
                    statusBadge = '<span class="badge badge-not-implemented">Not Impl.</span>';
                }
                
                const num = String(index + 1).padStart(2, '0');
                
                const fixedCols = 
                    '<td style="color: #8b949e; font-weight: 600;">' + num + '</td>' +
                    '<td>' + strat.id + '</td>' +
                    '<td>' + strat.timeframe + '</td>' +
                    '<td style="text-align: center; color: #58a6ff;">' + strat.symbols_count + '</td>';
                
                let commonCols = '';
                commonKeys.forEach(key => {
                    const value = strat[key] !== undefined ? strat[key] : 'N/A';
                    const display = key === 'order_amount' ? '$' + value : value;
                    commonCols += '<td>' + display + '</td>';
                });
                
                let extraCols = '';
                extraParamKeysArray.forEach(key => {
                    const value = strat[key];
                    const display = (value !== undefined && value !== 'N/A') ? value : '-';
                    extraCols += '<td>' + display + '</td>';
                });
                             
                return '<tr>' + fixedCols + commonCols + extraCols + '<td>' + statusBadge + '</td></tr>';
            }).join('');
        }
        
        // MODIFIED: Call new regime matrix functions
        await loadRegimeSizing();
        renderRegimeStrategyMatrix(data.strategies);
        
    } catch (error) {
        console.error('Error:', error);
    }
}

// =============================================================================
// GENERIC CHECKBOX FUNCTIONS (Used by Curves, Monthly, Correlation)
// =============================================================================

/**
 * Initialize strategy checkboxes for a tab
 * @param {string} containerId - ID of checkbox container div
 * @param {string} checkboxPrefix - Prefix for individual checkbox IDs (e.g., 'strat-')
 * @param {string} allCheckboxId - ID of "ALL STRATEGIES" checkbox
 * @returns {Promise<Array>} Array of strategies from API
 */
async function initStrategyCheckboxes(containerId, checkboxPrefix, allCheckboxId) {
    const res = await fetch('/api/bot-config');
    const data = await res.json();
    
    const strategies = data.strategies || [];
    
    const checkboxContainer = document.getElementById(containerId);
    const allCheckbox = checkboxContainer.querySelector(`#${allCheckboxId}`).parentElement;
    
    // Clear and re-add ALL checkbox
    checkboxContainer.innerHTML = '';
    checkboxContainer.appendChild(allCheckbox);
    
    // Create checkbox for each strategy
    strategies.forEach((strat) => {
        const div = document.createElement('div');
        div.className = 'checkbox-item';
        
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = checkboxPrefix + strat.id;
        checkbox.value = strat.id;
        checkbox.checked = (strat.status === 'ACTIVE' || strat.status === 'DEPRECATING');
        
        const label = document.createElement('label');
        label.htmlFor = checkboxPrefix + strat.id;
        
        const displayName = strat.id.replace(/^\d{2}_/, '');
        label.textContent = '[' + strat.number + '] ' + displayName + ' (' + strat.status + ')';
        
        // Color coding by status
        if (strat.status === 'ACTIVE') {
            label.style.color = '#ffffff';
            label.style.fontWeight = '600';
        } else if (strat.status === 'DEPRECATING') {
            label.style.color = '#d29922';
        } else {
            label.style.color = '#8b949e';
            label.style.fontStyle = 'italic';
        }
        
        div.appendChild(checkbox);
        div.appendChild(label);
        checkboxContainer.appendChild(div);
    });
    
    // Setup "ALL" checkbox toggle - use cloneNode to avoid duplicate listeners
    const allCheckboxInput = document.getElementById(allCheckboxId);
    const newAllCheckbox = allCheckboxInput.cloneNode(true);
    allCheckboxInput.parentNode.replaceChild(newAllCheckbox, allCheckboxInput);
    
    newAllCheckbox.addEventListener('change', function() {
        const isChecked = this.checked;
        document.querySelectorAll(`#${containerId} input[type="checkbox"]`).forEach(cb => {
            if (cb.id !== allCheckboxId) cb.checked = isChecked;
        });
    });
    
    return strategies;
}

/**
 * Get selected strategy IDs from a checkbox container
 * @param {string} containerId - ID of checkbox container div
 * @returns {Array<string>} Array of selected strategy IDs (excludes "ALL")
 */
function getSelectedStrategies(containerId) {
    const selected = [];
    document.querySelectorAll(`#${containerId} input[type="checkbox"]:checked`).forEach(cb => {
        if (cb.value !== 'ALL') {
            selected.push(cb.value);
        }
    });
    return selected;
}

// =============================================================================
// END GENERIC CHECKBOX FUNCTIONS
// =============================================================================

async function loadEquityTab() {
    try {
        allStrategiesList = await initStrategyCheckboxes(
            'strategy-checkboxes',
            'strat-',
            'strat-all'
        );
        
        await updateEquityChart();
        
    } catch (error) {
        console.error('Error loading equity tab:', error);
    }
}

async function initMonthlyTab() {
    try {
        allStrategiesList = await initStrategyCheckboxes(
            'monthly-strategy-checkboxes',
            'monthly-strat-',
            'monthly-strat-all'
        );
        
    } catch (error) {
        console.error('Error initializing monthly tab:', error);
    }
}

async function loadMonthlyAnalysis() {
    try {
        const selectedStrategies = getSelectedStrategies('monthly-strategy-checkboxes');
        
        if (selectedStrategies.length === 0) {
            alert('Please select at least one strategy');
            return;
        }
        
        const res = await fetch('/api/monthly-analysis?strategies=' + selectedStrategies.join(','));
        const data = await res.json();
        const container = document.getElementById('monthly-container');
        
        if (!data || data.length === 0) {
            container.innerHTML = '<div style="text-align: center; color: #8b949e; padding: 40px;">No data available for selected strategies</div>';
            return;
        }
        
        const html = '<div style="display: flex; flex-wrap: wrap; gap: 12px; padding: 10px 0;">' +
            data.map(row => {
                const profitPctClass = row.profit_pct >= 0 ? '#3fb950' : '#f85149';
                const profitUsdClass = row.profit_usd >= 0 ? '#3fb950' : '#f85149';
                const prefixPct = row.profit_pct >= 0 ? '+' : '';
                const prefixUsd = row.profit_usd >= 0 ? '+$' : '$';
                
                return '<div style="background: #1c2128; border: 1px solid #21262d; border-radius: 8px; padding: 15px 20px; min-width: 100px; text-align: center;">' +
                    '<div style="color: ' + profitPctClass + '; font-size: 18px; font-weight: 700; margin-bottom: 4px;">' + prefixPct + row.profit_pct.toFixed(1) + '%</div>' +
                    '<div style="color: ' + profitUsdClass + '; font-size: 14px; font-weight: 600; margin-bottom: 8px;">' + prefixUsd + row.profit_usd.toFixed(0) + '</div>' +
                    '<div style="color: #58a6ff; font-size: 13px; font-weight: 600; text-transform: uppercase;">' + row.month_name + '</div>' +
                    '</div>';
            }).join('') +
            '</div>';
        
        container.innerHTML = html;
        
    } catch (error) {
        console.error('Error loading monthly analysis:', error);
        document.getElementById('monthly-container').innerHTML = '<div style="text-align: center; color: #f85149; padding: 40px;">Error loading data</div>';
    }
}

async function loadComposeAnalysis() {
    try {
        document.querySelectorAll('.compose-checkbox').forEach(cb => cb.checked = false);
        
        // Show loading indicator
        const container = document.getElementById('compose-container');
        container.innerHTML = '<div style="text-align: center; color: #58a6ff; padding: 40px;"><div class="loading-spinner"></div><p style="margin-top: 15px;">Calculating combinations...</p></div>';
        document.getElementById('compose-plot-btn').style.display = 'none';
        
        const metric = document.getElementById('compose-metric').value;
        const dateParams = getComposeDateParams();
        const res = await fetch('/api/compose-analysis?metric=' + metric + dateParams);
        
        if (!res.ok) throw new Error('HTTP ' + res.status);
        
        const data = await res.json();
        
        if (!data || data.length === 0) {
            container.innerHTML = '<div style="text-align: center; color: #8b949e; padding: 40px;">No data available</div>';
            document.getElementById('compose-plot-btn').style.display = 'none';
            return;
        }
        
        function formatNumber(value, decimals = 2, suffix = '') {
            if (value === null || value === undefined) return '-';
            if (!isFinite(value)) return '-';
            return value.toFixed(decimals) + suffix;
        }
        
        function getRSquaredStyle(value) {
            if (!isFinite(value)) return '';
            if (value >= 0.9) return 'style="color: #6d28d9; text-shadow: 0 0 10px rgba(109, 40, 217, 0.8);"';
            if (value >= 0.7) return 'style="color: #3fb950;"';
            if (value >= 0.5) return 'style="color: #d29922;"';
            return 'style="color: #f85149;"';
        }
        
        function getMetricStyle(value, metric_type) {
            if (!isFinite(value)) return '';
            
            if (metric_type === 'profit_factor' || metric_type === 'sharpe') {
                if (value >= 2.0) return 'style="color: #6d28d9; text-shadow: 0 0 10px rgba(109, 40, 217, 0.8);"';
                if (value >= 1.5) return 'style="color: #3fb950;"';
                if (value >= 1.0) return 'style="color: #d29922;"';
                return 'style="color: #f85149;"';
            }
            return '';
        }
        
        const metricNames = {
            'num_trades': '#trades',
            'total_profit_pct': 'Profit_%',
            'total_profit_usd': 'Profit_$',
            'profit_factor': 'Profit Factor',
            'weekly_win_pct': 'Weekly_%',
            'win_rate': 'Win Rate',
            'max_dd': 'Max DD',
            'r_squared': 'R²',
            'sharpe_ratio': 'Sharpe'
        };
        
        const highlightCol = metricNames[metric];
        
        const html = '<table><thead><tr>' +
            '<th style="width: 40px;"></th>' +
            '<th>#</th>' +
            '<th>Combination</th>' +
            '<th' + (highlightCol === '#trades' ? ' style="font-weight: 900; color: #58a6ff;"' : '') + '>#trades</th>' +
            '<th' + (highlightCol === 'Profit_%' ? ' style="font-weight: 900; color: #58a6ff;"' : '') + '>Profit_%</th>' +
            '<th' + (highlightCol === 'Profit_$' ? ' style="font-weight: 900; color: #58a6ff;"' : '') + '>Profit_$</th>' +
            '<th' + (highlightCol === 'Profit Factor' ? ' style="font-weight: 900; color: #58a6ff;"' : '') + '>Profit Factor</th>' +
            '<th' + (highlightCol === 'Weekly_%' ? ' style="font-weight: 900; color: #58a6ff;"' : '') + '>Weekly_%</th>' +
            '<th' + (highlightCol === 'Win Rate' ? ' style="font-weight: 900; color: #58a6ff;"' : '') + '>Win Rate</th>' +
            '<th' + (highlightCol === 'Max DD' ? ' style="font-weight: 900; color: #58a6ff;"' : '') + '>Max DD</th>' +
            '<th' + (highlightCol === 'R²' ? ' style="font-weight: 900; color: #58a6ff;"' : '') + '>R²</th>' +
            '<th' + (highlightCol === 'Sharpe' ? ' style="font-weight: 900; color: #58a6ff;"' : '') + '>Sharpe</th>' +
            '</tr></thead><tbody>' +
            data.map((row, idx) => {
                const numTrades = row.num_trades || 0;
                const totalProfitPct = formatNumber(row.total_profit_pct, 1, '%');
                const profitUSD = formatNumber(row.total_profit_usd, 2);
                const profitFactor = formatNumber(row.profit_factor, 2);
                const weeklyWin = formatNumber(row.weekly_win_pct, 1, '%');
                const winRate = formatNumber(row.win_rate, 1, '%');
                const maxDD = formatNumber(row.max_dd, 2, '%');
                const rSquared = formatNumber(row.r_squared, 3);
                const sharpe = formatNumber(row.sharpe_ratio, 2);
                
                const profitPctClass = isFinite(row.total_profit_pct) ? (row.total_profit_pct >= 0 ? 'direction-long' : 'direction-short') : '';
                const profitUSDClass = isFinite(row.total_profit_usd) ? (row.total_profit_usd >= 0 ? 'direction-long' : 'direction-short') : '';
                
                const pfStyle = getMetricStyle(row.profit_factor, 'profit_factor');
                const sharpeStyle = getMetricStyle(row.sharpe_ratio, 'sharpe');
                const r2Style = getRSquaredStyle(row.r_squared);
                
                const ntStyle = highlightCol === '#trades' ? 'font-weight: 900;' : '';
                const tpStyle = highlightCol === 'Profit_%' ? 'font-weight: 900;' : '';
                const puStyle = highlightCol === 'Profit_$' ? 'font-weight: 900;' : '';
                const pfHighlight = highlightCol === 'Profit Factor' ? 'font-weight: 900;' : '';
                const wwStyle = highlightCol === 'Weekly_%' ? 'font-weight: 900;' : '';
                const wrStyle = highlightCol === 'Win Rate' ? 'font-weight: 900;' : '';
                const ddStyle = highlightCol === 'Max DD' ? 'font-weight: 900;' : '';
                const r2Highlight = highlightCol === 'R²' ? 'font-weight: 900;' : '';
                const sharpeHighlight = highlightCol === 'Sharpe' ? 'font-weight: 900;' : '';
                
                const prefixProfitPct = (isFinite(row.total_profit_pct) && row.total_profit_pct >= 0) ? '+' : '';
                const prefixProfitUSD = (isFinite(row.total_profit_usd) && row.total_profit_usd >= 0) ? '+$' : '$';
                
                return '<tr>' +
                    '<td><input type="checkbox" class="compose-checkbox" value="' + row.combination + '" onchange="checkComposeLimit()"></td>' +
                    '<td style="color: #8b949e; font-weight: 600;">' + (idx + 1) + '</td>' +
                    '<td style="color: #58a6ff; font-weight: 600;">' + row.combination + '</td>' +
                    '<td style="' + ntStyle + '">' + numTrades + '</td>' +
                    '<td class="' + profitPctClass + '" style="' + tpStyle + '">' + prefixProfitPct + totalProfitPct + '</td>' +
                    '<td class="' + profitUSDClass + '" style="' + puStyle + '">' + prefixProfitUSD + profitUSD + '</td>' +
                    '<td ' + pfStyle + ' style="' + pfHighlight + '">' + profitFactor + '</td>' +
                    '<td style="' + wwStyle + '">' + weeklyWin + '</td>' +
                    '<td style="' + wrStyle + '">' + winRate + '</td>' +
                    '<td style="color: #f85149; ' + ddStyle + '">' + maxDD + '</td>' +
                    '<td ' + r2Style + ' style="' + r2Highlight + '">' + rSquared + '</td>' +
                    '<td ' + sharpeStyle + ' style="' + sharpeHighlight + '">' + sharpe + '</td>' +
                    '</tr>';
            }).join('') +
            '</tbody></table>';
        
        container.innerHTML = html;
        document.getElementById('compose-plot-btn').style.display = 'block';
        
    } catch (error) {
        console.error('Error in compose:', error);
        document.getElementById('compose-plot-btn').style.display = 'none';
        document.getElementById('compose-container').innerHTML = 
            '<div style="text-align: center; color: #f85149; padding: 40px;">' +
            '<div style="font-size: 18px; margin-bottom: 10px;">❌ Error loading data</div>' +
            '<div style="font-size: 14px; color: #8b949e;">Check browser console (F12) for details</div>' +
            '<div style="font-size: 12px; color: #8b949e; margin-top: 10px;">Error: ' + error.message + '</div>' +
            '</div>';
    }
}

function checkComposeLimit() {
    const checkboxes = document.querySelectorAll('.compose-checkbox:checked');
    if (checkboxes.length > 3) {
        alert('Maximum 3 combinations can be selected for plotting');
        event.target.checked = false;
    }
}

let composeEquityChart = null;
let composeDrawdownChart = null;

function extractNumberFromId(strategyId) {
    if (!strategyId || typeof strategyId !== 'string') {
        return '??';
    }
    
    const match = strategyId.match(/^(\d{2})_/);
    if (match) {
        return match[1];
    }
    
    const fallbackMatch = strategyId.match(/^(\d+)/);
    if (fallbackMatch) {
        return fallbackMatch[1].padStart(2, '0');
    }
    
    return '??';
}

function findStrategyIdByNumber(number, strategiesList) {
    if (!strategiesList || strategiesList.length === 0) {
        return null;
    }
    
    const found = strategiesList.find(s => s.id && s.id.startsWith(number + '_'));
    return found ? found.id : null;
}

async function loadComposeCharts() {
    try {
        const checkboxes = document.querySelectorAll('.compose-checkbox:checked');
        const selectedCombinations = Array.from(checkboxes).map(cb => cb.value);
        
        if (selectedCombinations.length === 0) {
            alert('Please select at least one combination to plot');
            return;
        }
        
        if (selectedCombinations.length > 3) {
            alert('Maximum 3 combinations can be selected');
            return;
        }
        
        const dateParams = getComposeDateParams();
        
        const equityDatasets = [];
        const drawdownDatasets = [];
        let allDates = [];
        const allCombinationsData = [];
        
        for (let i = 0; i < selectedCombinations.length; i++) {
            const combination = selectedCombinations[i];
            const numbers = combination.split('+');
            
            const strategies = numbers
                .map(num => findStrategyIdByNumber(num, allStrategiesList))
                .filter(id => id);
            
            if (strategies.length === 0) {
                continue;
            }
            
            const res = await fetch('/api/equity-data?strategies=' + strategies.join(',') + dateParams);
            if (!res.ok) throw new Error('HTTP ' + res.status);
            const data = await res.json();
            
            if (!data.dates || !data.equity_pct || !data.drawdown_pct) {
                continue;
            }
            
            if (data.dates.length === 0) {
                continue;
            }
            
            allCombinationsData.push({
                combination: combination,
                dates: data.dates,
                equity_pct: data.equity_pct,
                drawdown_pct: data.drawdown_pct
            });
        }
        
        if (allCombinationsData.length === 0) {
            alert('⚠️ No valid data found for selected combinations.');
            return;
        }
        
        const allUniqueDates = new Set();
        allCombinationsData.forEach(combo => {
            combo.dates.forEach(date => allUniqueDates.add(date));
        });
        
        allDates = Array.from(allUniqueDates).sort();
        
        const colors = {
            equity: ['#3fb950', '#58a6ff', '#22d3ee'],
            equityAlpha: ['rgba(63, 185, 80, 0.1)', 'rgba(88, 166, 255, 0.1)', 'rgba(34, 211, 238, 0.1)'],
            drawdown: ['#f85149', '#ff6b6b', '#ff8787'],
            drawdownAlpha: ['rgba(248, 81, 73, 0.1)', 'rgba(255, 107, 107, 0.1)', 'rgba(255, 135, 135, 0.1)']
        };
        
        allCombinationsData.forEach((combo, colorIndex) => {
            const equityMap = {};
            const drawdownMap = {};
            
            combo.dates.forEach((date, idx) => {
                equityMap[date] = combo.equity_pct[idx];
                drawdownMap[date] = combo.drawdown_pct[idx];
            });
            
            const alignedEquity = [];
            const alignedDrawdown = [];
            let lastEquity = 0;
            let lastDrawdown = 0;
            
            allDates.forEach(date => {
                if (equityMap[date] !== undefined) {
                    lastEquity = equityMap[date];
                    lastDrawdown = drawdownMap[date];
                }
                alignedEquity.push(lastEquity);
                alignedDrawdown.push(lastDrawdown);
            });
            
            equityDatasets.push({
                label: combo.combination,
                data: alignedEquity,
                borderColor: colors.equity[colorIndex],
                backgroundColor: 'transparent',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.1,
                fill: false
            });
            
            drawdownDatasets.push({
                label: combo.combination,
                data: alignedDrawdown,
                borderColor: colors.drawdown[colorIndex],
                backgroundColor: colors.drawdownAlpha[colorIndex],
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.1,
                fill: true
            });
        });

        if (equityDatasets.length === 0) {
            alert('⚠️ No valid data found for selected combinations.');
            return;
        }
        
        if (allDates.length === 0) {
            alert('⚠️ No dates available in the data.');
            return;
        }
        
        if (composeEquityChart) {
            composeEquityChart.destroy();
            composeEquityChart = null;
        }
        if (composeDrawdownChart) {
            composeDrawdownChart.destroy();
            composeDrawdownChart = null;
        }
        
        const equityCanvas = document.getElementById('compose-equity-chart');
        const equityCtx = equityCanvas.getContext('2d');
        composeEquityChart = new Chart(equityCtx, {
            type: 'line',
            data: {
                labels: allDates,
                datasets: equityDatasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { 
                        display: true, 
                        position: 'top',
                        labels: { color: COLORS.white, font: { size: 14 } }
                    },
                    title: { 
                        display: true, 
                        text: 'Equity - Comparison',
                        color: CHART_DEFAULTS.titleColor,
                        font: { size: CHART_DEFAULTS.fontSize.title, weight: 'bold' }
                    }
                },
                scales: {
                    x: { 
                        ticks: { color: CHART_DEFAULTS.textColor, font: { size: CHART_DEFAULTS.fontSize.axis } }, 
                        grid: { color: CHART_DEFAULTS.gridColor, borderColor: CHART_DEFAULTS.borderColor, borderWidth: CHART_DEFAULTS.borderWidth }
                    },
                    y: { 
                        ticks: { 
                            color: CHART_DEFAULTS.textColor, 
                            font: { size: CHART_DEFAULTS.fontSize.axis },
                            callback: function(value) { return value.toFixed(1) + '%'; }
                        }, 
                        grid: { color: CHART_DEFAULTS.gridColor, borderColor: CHART_DEFAULTS.borderColor, borderWidth: CHART_DEFAULTS.borderWidth }
                    }
                }
            }
        });
        
        const drawdownCanvas = document.getElementById('compose-drawdown-chart');
        const drawdownCtx = drawdownCanvas.getContext('2d');
        composeDrawdownChart = new Chart(drawdownCtx, {
            type: 'line',
            data: {
                labels: allDates,
                datasets: drawdownDatasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { 
                        display: true, 
                        position: 'top',
                        labels: { color: COLORS.white, font: { size: 14 } }
                    },
                    title: { 
                        display: true, 
                        text: 'Drawdown - Comparison',
                        color: CHART_DEFAULTS.titleColor,
                        font: { size: CHART_DEFAULTS.fontSize.title, weight: 'bold' }
                    }
                },
                scales: {
                    x: { 
                        ticks: { color: CHART_DEFAULTS.textColor, font: { size: CHART_DEFAULTS.fontSize.axis } }, 
                        grid: { color: CHART_DEFAULTS.gridColor, borderColor: CHART_DEFAULTS.borderColor, borderWidth: CHART_DEFAULTS.borderWidth }
                    },
                    y: { 
                        reverse: true,
                        ticks: { 
                            color: CHART_DEFAULTS.textColor, 
                            font: { size: CHART_DEFAULTS.fontSize.axis },
                            callback: function(value) { return value.toFixed(1) + '%'; }
                        }, 
                        grid: { color: CHART_DEFAULTS.gridColor, borderColor: CHART_DEFAULTS.borderColor, borderWidth: CHART_DEFAULTS.borderWidth }
                    }
                }
            }
        });
        
    } catch (error) {
        console.error('Error loading compose charts:', error);
        alert('❌ Error loading charts:\n\n' + error.message);
    }
}

async function loadSymbolsAnalysis() {
    try {
        const res = await fetch('/api/symbols-analysis');
        const data = await res.json();
        const container = document.getElementById('symbols-container');
        
        if (!data || data.length === 0) {
            container.innerHTML = '<div style="text-align: center; color: #8b949e; padding: 40px;">No data available</div>';
            return;
        }
        
        const sortedData = data.sort((a, b) => b.Win_Pct - a.Win_Pct);
        
        function formatSlippage(value) {
            if (value === null || value === undefined) {
                return '<span style="color: #6b7280;">N/A</span>';
            }
            
            let color;
            
            // Positive slippage = better execution (always green)
            if (value > 0) {
                color = '#3fb950';
            } else {
                const absValue = Math.abs(value);
                if (absValue < SLIPPAGE_THRESHOLDS.warning) {
                    color = '#3fb950';
                } else if (absValue < SLIPPAGE_THRESHOLDS.critical) {
                    color = '#d29922';
                } else {
                    color = '#f85149';
                }
            }
            
            const prefix = value >= 0 ? '+' : '';
            return `<span style="color: ${color}; font-weight: 600;">${prefix}${value.toFixed(2)}%</span>`;
        }
        
        const html = '<table><thead><tr><th>Symbol</th><th>Total Trades</th><th>Win %</th><th>Total Profit</th><th>Avg Profit</th><th>Slippage Total</th><th>Slippage L30</th></tr></thead><tbody>' +
            sortedData.map(s => {
                const profitClass = s.Total_Profit >= 0 ? 'direction-long' : 'direction-short';
                const avgProfitClass = s.Avg_Profit >= 0 ? 'direction-long' : 'direction-short';
                
                return '<tr>' +
                    '<td>' + s.Symbol + '</td>' +
                    '<td>' + s.Total_Trades + '</td>' +
                    '<td>' + s.Win_Pct.toFixed(1) + '%</td>' +
                    '<td class="' + profitClass + '">' + (s.Total_Profit >= 0 ? '+' : '') + '$' + s.Total_Profit.toFixed(2) + '</td>' +
                    '<td class="' + avgProfitClass + '">' + (s.Avg_Profit >= 0 ? '+' : '') + '$' + s.Avg_Profit.toFixed(2) + '</td>' +
                    '<td>' + formatSlippage(s.Slippage_Total) + '</td>' +
                    '<td>' + formatSlippage(s.Slippage_L30) + '</td>' +
                    '</tr>';
            }).join('') +
            '</tbody></table>';
        container.innerHTML = html;
    } catch (error) {
        console.error('Error loading symbols analysis:', error);
        document.getElementById('symbols-container').innerHTML = '<div style="text-align: center; color: #f85149; padding: 40px;">Error loading data</div>';
    }
}
async function loadWeekDayAnalysis() {
    try {
        const res = await fetch('/api/weekday-analysis');
        const data = await res.json();
        const container = document.getElementById('weekday-container');
        
        if (!data || data.length === 0) {
            container.innerHTML = '<div style="text-align: center; color: #8b949e; padding: 40px;">No data available</div>';
            return;
        }
        
        const html = '<table><thead><tr><th>Day</th><th>Total Trades</th><th>Win %</th><th>Total Profit</th><th>Avg Profit</th></tr></thead><tbody>' +
            data.map(d => {
                const profitClass = d.Total_Profit >= 0 ? 'direction-long' : 'direction-short';
                const avgProfitClass = d.Avg_Profit >= 0 ? 'direction-long' : 'direction-short';
                return '<tr><td>' + d.Day + '</td><td>' + d.Total_Trades + '</td><td>' + d.Win_Pct.toFixed(1) + '%</td><td class="' + profitClass + '">' + (d.Total_Profit >= 0 ? '+' : '') + '$' + d.Total_Profit.toFixed(2) + '</td><td class="' + avgProfitClass + '">' + (d.Avg_Profit >= 0 ? '+' : '') + '$' + d.Avg_Profit.toFixed(2) + '</td></tr>';
            }).join('') +
            '</tbody></table>';
        container.innerHTML = html;
    } catch (error) {
        console.error('Error loading weekday analysis:', error);
        document.getElementById('weekday-container').innerHTML = '<div style="text-align: center; color: #f85149; padding: 40px;">Error loading data</div>';
    }
}

// =============================================================================
// REGIME ANALYTICS
// =============================================================================

async function loadRegimeAnalytics() {
    try {
        const res = await fetch('/api/analytics/regime');
        
        if (!res.ok) {
            throw new Error('HTTP ' + res.status);
        }
        
        const response = await res.json();
        const container = document.getElementById('regime-analytics-container');
        
        if (!response.success || !response.data || Object.keys(response.data).length === 0) {
            container.innerHTML = '<div style="text-align: center; color: #8b949e; padding: 40px;">' +
                (response.error || 'No regime data available yet') +
                '</div>';
            return;
        }
        
        const data = response.data;
        
        // Sort regimes by priority: trending > ranging > volatile > unknown
        const regimeOrder = ['trending', 'ranging', 'volatile', 'unknown'];
        const sortedRegimes = Object.keys(data).sort((a, b) => {
            const indexA = regimeOrder.indexOf(a);
            const indexB = regimeOrder.indexOf(b);
            return (indexA === -1 ? 999 : indexA) - (indexB === -1 ? 999 : indexB);
        });
        
        // Colors for each regime - UPDATED
        const regimeColors = {
            'trending': { bar: '#58a6ff', text: '#58a6ff', bg: 'rgba(88, 166, 255, 0.1)' },        // AZUL
            'ranging': { bar: '#9ca3af', text: '#9ca3af', bg: 'rgba(156, 163, 175, 0.1)' },        // GRIS CLARO
            'volatile': { bar: '#f85149', text: '#f85149', bg: 'rgba(248, 81, 73, 0.1)' },        // ROJO
            'unknown': { bar: '#8b949e', text: '#8b949e', bg: 'rgba(139, 148, 158, 0.1)' }
        };
        
        let html = '<div style="display: flex; flex-direction: column; gap: 20px;">';
        
        sortedRegimes.forEach(regime => {
            const stats = data[regime];
            const colors = regimeColors[regime] || regimeColors['unknown'];
            
            const winrateWidth = Math.round(stats.winrate);
            const pnlClass = stats.pnl >= 0 ? 'direction-long' : 'direction-short';
            const pnlPrefix = stats.pnl >= 0 ? '+$' : '$';
            
            html += `
                <div style="background: ${colors.bg}; border: 1px solid ${colors.bar}; border-radius: 8px; padding: 20px;">
                    <!-- Header -->
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                        <div style="font-size: 24px; font-weight: 700; text-transform: uppercase; color: ${colors.text};">
                            ${regime}
                        </div>
                        <div style="font-size: 24px; color: #8b949e;">
                            ${stats.trades}/${stats.total_trades} trades
                        </div>
                    </div>
                    
                    <!-- Win Rate Bar -->
                    <div style="margin-bottom: 15px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <span style="color: #8b949e; font-size: 21px; font-weight: 600;">Win Rate</span>
                            <span style="color: ${colors.text}; font-size: 21px; font-weight: 700;">${stats.winrate.toFixed(1)}%</span>
                        </div>
                        <div style="background: #21262d; height: 20px; border-radius: 4px; overflow: hidden;">
                            <div style="background: ${colors.bar}; height: 100%; width: ${winrateWidth}%; transition: width 0.3s ease;"></div>
                        </div>
                    </div>
                    
                    <!-- P&L -->
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #8b949e; font-size: 21px; font-weight: 600;">P&L</span>
                        <span class="${pnlClass}" style="font-size: 27px; font-weight: 700;">${pnlPrefix}${stats.pnl.toFixed(2)}</span>
                    </div>
                </div>
            `;
        });
        
        html += '</div>';
        
        container.innerHTML = html;
        
    } catch (error) {
        console.error('Error loading regime analytics:', error);
        document.getElementById('regime-analytics-container').innerHTML = 
            '<div style="text-align: center; color: #f85149; padding: 40px;">Error loading regime data</div>';
    }
}

// =============================================================================
// REGIME ANALYTICS MODE SELECTOR
// =============================================================================

function setRegimeAnalyticsMode(mode) {
    currentRegimeAnalyticsMode = mode;
    
    // Update button states
    document.querySelectorAll('#equity-subtab-regime .view-selector .view-btn').forEach(btn => {
        btn.classList.remove('active');
        if ((mode === 'regime' && btn.textContent.includes('Market Regime')) ||
            (mode === 'direction' && btn.textContent.includes('Market Direction'))) {
            btn.classList.add('active');
        }
    });
    
    // Load corresponding data
    if (mode === 'regime') {
        loadRegimeAnalytics();
    } else {
        loadMarketDirectionAnalytics();
    }
}

// =============================================================================
// MARKET DIRECTION ANALYTICS
// =============================================================================

async function loadMarketDirectionAnalytics() {
    try {
        const res = await fetch('/api/analytics/market-direction');
        
        if (!res.ok) {
            throw new Error('HTTP ' + res.status);
        }
        
        const response = await res.json();
        const container = document.getElementById('regime-analytics-container');
        
        if (!response.success || !response.data || Object.keys(response.data).length === 0) {
            container.innerHTML = '<div style="text-align: center; color: #8b949e; padding: 40px;">' +
                (response.error || 'No market direction data available yet') +
                '</div>';
            return;
        }
        
        const data = response.data;
        
        // Sort directions: uptrend > downtrend > unknown
        const directionOrder = ['uptrend', 'dwtrend', 'unknown'];
        const sortedDirections = Object.keys(data).sort((a, b) => {
            const indexA = directionOrder.indexOf(a);
            const indexB = directionOrder.indexOf(b);
            return (indexA === -1 ? 999 : indexA) - (indexB === -1 ? 999 : indexB);
        });
        
        // Colors for each direction
        const directionColors = {
            'uptrend': { bar: '#3fb950', text: '#3fb950', bg: 'rgba(63, 185, 80, 0.1)' },      // Green
            'dwtrend': { bar: '#f85149', text: '#f85149', bg: 'rgba(248, 81, 73, 0.1)' },   // Red
            'unknown': { bar: '#8b949e', text: '#8b949e', bg: 'rgba(139, 148, 158, 0.1)' }
        };
        
        let html = '<div style="display: flex; flex-direction: column; gap: 20px;">';
        
        sortedDirections.forEach(direction => {
            const stats = data[direction];
            const colors = directionColors[direction] || directionColors['unknown'];
            
            const winrateWidth = Math.round(stats.winrate);
            const pnlClass = stats.pnl >= 0 ? 'direction-long' : 'direction-short';
            const pnlPrefix = stats.pnl >= 0 ? '+$' : '$';
            
            html += `
                <div style="background: ${colors.bg}; border: 1px solid ${colors.bar}; border-radius: 8px; padding: 20px;">
                    <!-- Header -->
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                        <div style="font-size: 24px; font-weight: 700; text-transform: uppercase; color: ${colors.text};">
                            ${direction}
                        </div>
                        <div style="font-size: 24px; color: #8b949e;">
                            ${stats.trades}/${stats.total_trades} trades
                        </div>
                    </div>
                    
                    <!-- Win Rate Bar -->
                    <div style="margin-bottom: 15px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <span style="color: #8b949e; font-size: 21px; font-weight: 600;">Win Rate</span>
                            <span style="color: ${colors.text}; font-size: 21px; font-weight: 700;">${stats.winrate.toFixed(1)}%</span>
                        </div>
                        <div style="background: #21262d; height: 20px; border-radius: 4px; overflow: hidden;">
                            <div style="background: ${colors.bar}; height: 100%; width: ${winrateWidth}%; transition: width 0.3s ease;"></div>
                        </div>
                    </div>
                    
                    <!-- P&L -->
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #8b949e; font-size: 21px; font-weight: 600;">P&L</span>
                        <span class="${pnlClass}" style="font-size: 27px; font-weight: 700;">${pnlPrefix}${stats.pnl.toFixed(2)}</span>
                    </div>
                </div>
            `;
        });
        
        html += '</div>';
        
        container.innerHTML = html;
        
    } catch (error) {
        console.error('Error loading market direction analytics:', error);
        document.getElementById('regime-analytics-container').innerHTML = 
            '<div style="text-align: center; color: #f85149; padding: 40px;">Error loading market direction data</div>';
    }
}

// =============================================================================
// END REGIME ANALYTICS
// =============================================================================
// =============================================================================
// REGIME STRATEGY BREAKDOWN
// =============================================================================

function clearRegimeBreakdownDates() {
    document.getElementById('regime-breakdown-date-from').value = '';
    document.getElementById('regime-breakdown-date-to').value = '';
}

function getRegimeBreakdownDateParams() {
    const dateFrom = document.getElementById('regime-breakdown-date-from').value;
    const dateTo = document.getElementById('regime-breakdown-date-to').value;
    let params = '';
    if (dateFrom) params += '&date_from=' + dateFrom;
    if (dateTo) params += '&date_to=' + dateTo;
    return params;
}

async function loadRegimeStrategyBreakdown() {
    try {
        const dateParams = getRegimeBreakdownDateParams();
        const res = await fetch('/api/regime/strategy-breakdown?' + dateParams);
        
        if (!res.ok) {
            throw new Error('HTTP ' + res.status);
        }
        
        const response = await res.json();
        const container = document.getElementById('regime-breakdown-container');
        
        if (!response.success || !response.data || response.data.length === 0) {
            container.innerHTML = '<div style="text-align: center; color: #8b949e; padding: 40px;">' +
                (response.error || 'No data available for selected date range') +
                '</div>';
            return;
        }
        
        const data = response.data;
        
        // Build table
        let html = '<table><thead><tr>' +
            '<th>#</th>' +
            '<th>Strategy</th>' +
            '<th>Total Trades</th>' +
            '<th>Win %</th>' +
            '<th>Profit</th>' +
            '<th>TRENDING</th>' +
            '<th>RANGING</th>' +
            '<th>VOLATILE</th>' +
            '<th>UPTREND</th>' +
            '<th>DOWNTREND</th>' +
            '</tr></thead><tbody>';
        
        data.forEach(row => {
            const num = String(row.number).padStart(2, '0');
            const profitClass = row.profit >= 0 ? 'direction-long' : 'direction-short';
            const profitPrefix = row.profit >= 0 ? '+$' : '$';
            
            // Helper to format regime/direction cells
            // Helper to format regime/direction cells
            function formatCell(stats, globalWinRate) {
                if (stats.trades === 0) {
                    return '<span style="color: #6b7280;">-</span>';
                }
                
                let arrowColor = '#6b7280'; // Default grey
                let arrow = '=';
                
                if (stats.win_pct > globalWinRate) {
                    arrowColor = '#3fb950'; // Green
                    arrow = '↑';
                } else if (stats.win_pct < globalWinRate) {
                    arrowColor = '#f85149'; // Red
                    arrow = '↓';
                }
                
            return '<span style="color: #58a6ff;">' + stats.trades + ' / ' + stats.win_pct.toFixed(1) + '%</span>' +
                   ' <span style="color: ' + arrowColor + '; font-weight: 700; font-size: 24px;">' + arrow + '</span>';
            }
            
            html += '<tr>' +
                '<td style="color: #8b949e; font-weight: 600;">' + num + '</td>' +
                '<td>' + row.strategy + '</td>' +
                '<td style="text-align: center;">' + row.total_trades + '</td>' +
                '<td style="text-align: center;">' + row.win_rate.toFixed(1) + '%</td>' +
                '<td class="' + profitClass + '">' + profitPrefix + row.profit.toFixed(2) + '</td>' +
                '<td style="text-align: center;">' + formatCell(row.trending, row.win_rate) + '</td>' +
                '<td style="text-align: center;">' + formatCell(row.ranging, row.win_rate) + '</td>' +
                '<td style="text-align: center;">' + formatCell(row.volatile, row.win_rate) + '</td>' +
                '<td style="text-align: center;">' + formatCell(row.uptrend, row.win_rate) + '</td>' +
                '<td style="text-align: center;">' + formatCell(row.downtrend, row.win_rate) + '</td>' +
                '</tr>';
        });
        
        html += '</tbody></table>';
        container.innerHTML = html;
        
    } catch (error) {
        console.error('Error loading regime strategy breakdown:', error);
        document.getElementById('regime-breakdown-container').innerHTML = 
            '<div style="text-align: center; color: #f85149; padding: 40px;">Error loading data</div>';
    }
}

// =============================================================================
// END REGIME STRATEGY BREAKDOWN
// =============================================================================
async function updateEquityChart() {
    try {
        const selectedStrategies = getSelectedStrategies('strategy-checkboxes');
        
        if (selectedStrategies.length === 0) {
            alert('Please select at least one strategy');
            return;
        }
        
        const dateParams = getCurvesDateParams();
        const res = await fetch('/api/equity-data?strategies=' + selectedStrategies.join(',') + dateParams);
        const data = await res.json();
        
        if (!data.dates || data.dates.length === 0) {
            const strategiesStr = selectedStrategies.join(', ');
            alert(`No trades found for:\n${strategiesStr}\n\n${data.message || 'These strategies have no closed trades yet.'}`);
            return;
        }
        
        // Fetch BTC history with same date filters
        const btcRes = await fetch('/api/btc/history?timeframe=1Dutc' + dateParams);
        const btcData = await btcRes.json();
        
        document.getElementById('equity-metrics').style.display = 'block';
        document.getElementById('metric-num-trades').textContent = data.num_trades || 0;
        
        const profitPct = ((data.total_profit_usd / data.capital_assigned) * 100) || 0;
        document.getElementById('metric-profit-pct').textContent = (profitPct >= 0 ? '+' : '') + profitPct.toFixed(2) + '%';
        
        document.getElementById('metric-profit-usd').textContent = '$' + (data.total_profit_usd || 0);
        document.getElementById('metric-profit-factor').textContent = data.profit_factor || '-';
        document.getElementById('metric-weekly-win').textContent = (data.weekly_win_pct || 0) + '%';
        document.getElementById('metric-win-rate').textContent = (data.win_rate || 0) + '%';
        document.getElementById('metric-max-dd').textContent = (data.max_dd || 0) + '%';
        document.getElementById('metric-r-squared').textContent = data.r_squared || 0;
        document.getElementById('metric-sharpe').textContent = (data.sharpe_ratio || 0);
        
        const pctElement = document.getElementById('metric-profit-pct');
        applyMetricColor(pctElement, profitPct, 'positiveNegative');
        
        const puElement = document.getElementById('metric-profit-usd');
        applyMetricColor(puElement, data.total_profit_usd, 'positiveNegative');
        
        const pfElement = document.getElementById('metric-profit-factor');
        applyMetricColor(pfElement, data.profit_factor, 'profitFactor');
        
        const r2Element = document.getElementById('metric-r-squared');
        applyMetricColor(r2Element, data.r_squared, 'rSquared');
        
        const sharpeElement = document.getElementById('metric-sharpe');
        applyMetricColor(sharpeElement, data.sharpe_ratio, 'sharpe');
        
        if (equityChart) equityChart.destroy();
        if (drawdownChart) drawdownChart.destroy();
        
        const finalEquity = data.equity_pct[data.equity_pct.length - 1] || 0;
        const equityColor = finalEquity >= 0 ? COLORS.equityPositive : COLORS.equityNegative;
        
        // Prepare datasets
        const datasets = [{
            label: 'Equity (%)',
            data: data.equity_pct,
            borderColor: equityColor,
            backgroundColor: 'transparent',
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.1,
            yAxisID: 'y'
        }];
        
        // Add BTC dataset if data available
        // Add BTC dataset if data available (WITH DATE ALIGNMENT)
        if (btcData.success && btcData.dates && btcData.dates.length > 0) {
            // Normalize BTC dates: "YYYY-MM-DD HH:MM:SS" -> "YYYY-MM-DD"
            const btcDatesMap = {};
            btcData.dates.forEach((dateStr, idx) => {
                const normalizedDate = dateStr.split(' ')[0]; // Extract YYYY-MM-DD
                btcDatesMap[normalizedDate] = btcData.prices[idx];
            });
            
            // Align BTC prices with equity dates
            const alignedBtcPrices = data.dates.map(equityDate => {
                return btcDatesMap[equityDate] || null; // null if no match
            });
            
            // Only add if we have at least some overlap
            const validPrices = alignedBtcPrices.filter(p => p !== null);
            if (validPrices.length > 0) {
                datasets.push({
                    label: 'BTC Price',
                    data: alignedBtcPrices,
                    borderColor: '#f59e0b',
                    backgroundColor: 'transparent',
                    borderWidth: 1,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    tension: 0.1,
                    yAxisID: 'y2',
                    spanGaps: true  // Draw line across null values
                });
            }
        }
        const ctxEquity = document.getElementById('equityChart').getContext('2d');
        equityChart = new Chart(ctxEquity, {
            type: 'line',
            data: {
                labels: data.dates,
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                plugins: {
                    legend: { 
                        display: true,
                        position: 'top',
                        labels: {
                            color: COLORS.white,
                            font: { size: 14 }
                        }
                    },
                    title: { 
                        display: true, 
                        text: 'Equity Curve - ' + selectedStrategies.length + ' strategies selected',
                        color: CHART_DEFAULTS.titleColor,
                        font: { size: CHART_DEFAULTS.fontSize.title, weight: 'bold' }
                    }
                },
                scales: {
                    x: { 
                        ticks: { color: CHART_DEFAULTS.textColor, font: { size: CHART_DEFAULTS.fontSize.axis } }, 
                        grid: { 
                            color: CHART_DEFAULTS.gridColor,
                            drawBorder: true,
                            borderColor: CHART_DEFAULTS.borderColor,
                            borderWidth: CHART_DEFAULTS.borderWidth
                        } 
                    },
                    y: { 
                        type: 'linear',
                        display: true,
                        position: 'left',
                        ticks: { 
                            color: CHART_DEFAULTS.textColor,
                            font: { size: CHART_DEFAULTS.fontSize.axis },
                            callback: function(value) { return value.toFixed(1) + '%'; }
                        }, 
                        grid: { 
                            color: CHART_DEFAULTS.gridColor,
                            drawBorder: true,
                            borderColor: CHART_DEFAULTS.borderColor,
                            borderWidth: CHART_DEFAULTS.borderWidth
                        }
                    },
                    y2: {
                        type: 'linear',
                        display: datasets.length > 1,
                        position: 'right',
                        ticks: {
                            color: '#f59e0b',
                            font: { size: CHART_DEFAULTS.fontSize.axis },
                            callback: function(value) { return '$' + value.toLocaleString(); }
                        },
                        grid: {
                            drawOnChartArea: false
                        }
                    }
                }
            }
        });
        
        const ctxDD = document.getElementById('drawdownChart').getContext('2d');
        drawdownChart = new Chart(ctxDD, {
            type: 'line',
            data: {
                labels: data.dates,
                datasets: [{
                    label: 'Drawdown (%)',
                    data: data.drawdown_pct,
                    borderColor: COLORS.drawdownRed,
                    backgroundColor: COLORS.drawdownRedAlpha,
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.1,
                    fill: true
                }]
            },
            options: getBaseChartConfig('Drawdown - ' + selectedStrategies.length + ' strategies selected', true)
        });
        
    } catch (error) {
        console.error('Error updating equity chart:', error);
        alert('Error loading equity data: ' + error.message);
    }
}

async function loadData() {
    if (isLoadingData) return;
    isLoadingData = true;
    try {
        const statusRes = await fetch('/api/status');
        if (!statusRes.ok) throw new Error('HTTP ' + statusRes.status);
        const status = await statusRes.json();
        
        requestAnimationFrame(() => {
            const totalPosEl = document.getElementById('total-positions');
            if (totalPosEl) totalPosEl.textContent = status.total_positions || 0;
            
            const totalProfit = status.total_profit || 0;
            const profitEl = document.getElementById('total-profit');
            if (profitEl) {
                profitEl.textContent = '$' + totalProfit.toFixed(2);
                profitEl.className = 'stat-value ' + (totalProfit >= 0 ? 'positive' : 'negative');
            }
            
            const openPnl = status.open_pnl || 0;
            const openPnlEl = document.getElementById('open-pnl');
            if (openPnlEl) {
                openPnlEl.textContent = '$' + openPnl.toFixed(2);
                openPnlEl.className = 'stat-value ' + (openPnl >= 0 ? 'positive' : 'negative');
            }
            
            const profitPct = status.profit_pct || 0;
            const profitPctEl = document.getElementById('profit-pct');
            if (profitPctEl) {
                profitPctEl.textContent = (profitPct >= 0 ? '+' : '') + profitPct.toFixed(2) + '%';
                profitPctEl.className = 'stat-value ' + (profitPct >= 0 ? 'positive' : 'negative');
            }
            
            const tradesNumEl = document.getElementById('trades-num');
            if (tradesNumEl) tradesNumEl.textContent = status.num_trades || 0;
            
            const tradesPct = status.trades_pct || 0;
            const tradesPctEl = document.getElementById('trades-pct');
            if (tradesPctEl) tradesPctEl.textContent = tradesPct.toFixed(1) + '%';
            
            const btcPrice = status.btc_price || 0;
            const btcPriceEl = document.getElementById('btc-price');
            if (btcPriceEl) btcPriceEl.textContent = '$' + btcPrice.toLocaleString();
        });
        
        // Load exposure data with dynamic limits from backend
        try {
            const exposureRes = await fetch('/api/risk/exposure');
            if (exposureRes.ok) {
                const exposureData = await exposureRes.json();
                if (exposureData.success) {
                    // Update global limits from backend
                    if (exposureData.limits) {
                        MAX_GROSS_EXPOSURE = exposureData.limits.max_gross;
                        MAX_NET_EXPOSURE = exposureData.limits.max_net;
                    }
                    
                    const grossPct = exposureData.metrics.gross_exposure_pct;
                    const netPct = exposureData.metrics.net_exposure_pct;
                    
                    // Update header cards (always blue)
                    const grossEl = document.getElementById('exposure-gross');
                    const netEl = document.getElementById('exposure-net');
                    
                    if (grossEl) {
                        grossEl.textContent = grossPct.toFixed(1) + '%';
                        grossEl.style.color = '#58a6ff';  // Blue
                    }
                    
                    if (netEl) {
                        netEl.textContent = (netPct >= 0 ? '+' : '') + netPct.toFixed(1) + '%';
                        netEl.style.color = '#58a6ff';  // Blue
                    }
                }
            }
        } catch (error) {
            console.error('Error loading exposure:', error);
        }
        
        const posRes = await fetch('/api/positions');
        if (!posRes.ok) throw new Error('HTTP ' + posRes.status);
        const positions = await posRes.json();
        
        requestAnimationFrame(() => {
            if (currentTab === 'positions') renderPositions(positions);
        });
        
        const tradesRes = await fetch('/api/trades/recent');
        if (!tradesRes.ok) throw new Error('HTTP ' + tradesRes.status);
        const trades = await tradesRes.json();
        
        requestAnimationFrame(() => {
            const tradesBody = document.getElementById('trades-body');
            if (!trades || trades.length === 0) {
                tradesBody.innerHTML = '<tr><td colspan="10" style="text-align: center;">No trades</td></tr>';
            } else {
                tradesBody.innerHTML = trades.map(trade => {
                    const profitClass = trade.PROFIT >= 0 ? 'direction-long' : 'direction-short';
                    let reasonBadge = '';
                    if (trade.REASON_OUT === 'TP') {
                        reasonBadge = '<span class="badge badge-tp">TP</span>';
                    } else if (trade.REASON_OUT === 'SL') {
                        reasonBadge = '<span class="badge badge-sl">SL</span>';
                    } else {
                        reasonBadge = '<span class="badge badge-timeout">TIMEOUT</span>';
                    }
                    return '<tr>' +
                        '<td>' + new Date(trade.CLOSE_AT).toISOString().replace('T', ' ').substring(0, 19) + '</td>' +
                        '<td>' + trade.STRATEGY + '</td>' +
                        '<td>' + trade.SYMBOL + '</td>' +
                        '<td class="direction-' + trade.DIRECTION.toLowerCase() + '">' + trade.DIRECTION + '</td>' +
                        '<td>$' + parseFloat(trade.USDT_AMOUNT || 0).toFixed(2) + '</td>' +
                        '<td>$' + parseFloat(trade.PRICE_ENTRY || 0).toFixed(2) + '</td>' +
                        '<td>$' + parseFloat(trade.PRICE_CLOSE || 0).toFixed(2) + '</td>' +
                        '<td class="' + profitClass + '">' + (trade.PROFIT >= 0 ? '+' : '') + '$' + trade.PROFIT.toFixed(2) + '</td>' +
                        '<td class="' + profitClass + '">' + (trade.PROFIT_PCT >= 0 ? '+' : '') + trade.PROFIT_PCT.toFixed(2) + '%</td>' +
                        '<td>' + reasonBadge + '</td>' +
                        '</tr>';
                }).join('');
            }
        });
        
        // Load regime data (non-blocking)
        loadRegimeData().catch(console.error);
        
    } catch (error) {
        console.error('Error:', error);
    } finally {
        isLoadingData = false;
    }
}

function toggleAutoScroll() {
    autoScroll = !autoScroll;
    document.getElementById('auto-scroll-btn').textContent = autoScroll ? 'Auto' : 'Manual';
}

function clearLogs() {
    if (confirm('Clear all logs?')) {
        document.getElementById('logs-content').innerHTML = '<div class="log-line default">Logs cleared</div>';
    }
}

const POLLING_CONFIG = {
    data: {
        active: 5000,
        inactive: 90000
    },
    logs: {
        active: 2000,
        inactive: 90000
    }
};

let dataInterval, logsInterval;

function updateClock() {
    const now = new Date();
    const hours = String(now.getHours()).padStart(2, '0');
    const minutes = String(now.getMinutes()).padStart(2, '0');
    const seconds = String(now.getSeconds()).padStart(2, '0');
    const clockEl = document.getElementById('live-clock');
    if (clockEl) {
        clockEl.textContent = `${hours}:${minutes}:${seconds}`;
    }
}

function startPolling() {
    stopPolling();
    
    loadData();
    loadLogs();
    updateClock();
    
    const isActive = !document.hidden;
    const dataDelay = isActive ? POLLING_CONFIG.data.active : POLLING_CONFIG.data.inactive;
    const logsDelay = isActive ? POLLING_CONFIG.logs.active : POLLING_CONFIG.logs.inactive;
    
    dataInterval = setInterval(() => loadData().catch(console.error), dataDelay);
    logsInterval = setInterval(() => loadLogs().catch(console.error), logsDelay);
    setInterval(updateClock, 1000);
}

function stopPolling() {
    if (dataInterval) clearInterval(dataInterval);
    if (logsInterval) clearInterval(logsInterval);
}

document.addEventListener('visibilitychange', () => {
    if (document.hidden) stopPolling();
    else startPolling();
});

const savedView = 'compact';
currentPositionsView = savedView;

async function initializeDashboard() {
    document.querySelectorAll('.view-btn').forEach(btn => {
        if (btn.textContent.toLowerCase().includes(savedView)) btn.classList.add('active');
        else btn.classList.remove('active');
    });
    
    const backendReady = await waitForBackend();
    await loadQualityThresholds();
    await loadBotConfig();
    startPolling();
}

// =============================================================================
// CORRELATION ANALYSIS
// =============================================================================

async function initCorrelationTab() {
    try {
        allStrategiesList = await initStrategyCheckboxes(
            'correlation-strategy-checkboxes',
            'correlation-strat-',
            'correlation-strat-all'
        );
        
    } catch (error) {
        console.error('Error loading correlation tab:', error);
    }
}

function getSelectedCorrelationStrategies() {
    return getSelectedStrategies('correlation-strategy-checkboxes');
}


function setCorrelationMetric(metric) {
    currentCorrelationMetric = metric;
    
    // Update button states
    const buttons = document.querySelectorAll('#equity-subtab-correlation .view-selector .view-btn');
    buttons.forEach(btn => {
        btn.classList.remove('active');
        if ((metric === 'profit' && btn.textContent === 'Profit') ||
            (metric === 'drawdown' && btn.textContent === 'Drawdown')) {
            btn.classList.add('active');
        }
    });
}
async function updateCorrelationAnalysis() {
    const selected = getSelectedCorrelationStrategies();
    
    if (selected.length < 2) {
        alert('Please select at least 2 strategies to calculate correlation');
        return;
    }
    
    // Show loading
    document.getElementById('correlation-matrix-container').innerHTML = '<p style="text-align: center; padding: 40px; color: #8b949e;">Calculating correlation...</p>';
    document.getElementById('high-corr-pairs').innerHTML = '<p style="text-align: center; padding: 20px; color: #8b949e;">Calculating...</p>';
    
    try {
        const res = await fetch('/api/correlation-matrix', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                strategies: selected,
                metric: currentCorrelationMetric
            })
        });
        
        const data = await res.json();
        
        if (data.error) {
            document.getElementById('correlation-matrix-container').innerHTML = `<p style="color: #f85149; text-align: center; padding: 40px;">${data.error}</p>`;
            document.getElementById('high-corr-pairs').innerHTML = `<p style="color: #f85149; text-align: center; padding: 20px;">${data.error}</p>`;
            return;
        }
        
        renderCorrelationMatrix(data.matrix, data.strategies);
        renderHighCorrelationPairs(data.high_corr_pairs);
        
    } catch (error) {
        console.error('Error calculating correlation:', error);
        document.getElementById('correlation-matrix-container').innerHTML = '<p style="color: #f85149; text-align: center; padding: 40px;">Error calculating correlation</p>';
    }
}

function renderCorrelationMatrix(matrix, strategies) {
    const container = document.getElementById('correlation-matrix-container');
    
    let html = '<table class="correlation-table"><thead><tr><th></th>';
    
    // Header row - SOLO NÚMEROS
    strategies.forEach(strat => {
        const number = strat.split('_')[0];  // Solo el número
        html += `<th>${number}</th>`;
    });
    html += '</tr></thead><tbody>';
    
    // Data rows - SOLO NÚMEROS
    strategies.forEach(strat1 => {
        const number1 = strat1.split('_')[0];  // Solo el número
        html += `<tr><td>${number1}</td>`;
        
        strategies.forEach(strat2 => {
            const corr = matrix[strat1][strat2];
            const color = getCorrelationColor(corr);
            const displayValue = corr === 1.0 ? '1.00' : corr.toFixed(2);
            
            html += `<td class="correlation-cell" style="background-color: ${color};" title="${strat1} vs ${strat2}: ${displayValue}">${displayValue}</td>`;
        });
        
        html += '</tr>';
    });
    
    html += '</tbody></table>';
    container.innerHTML = html;
}

function getCorrelationColor(corr) {
    // High correlation (>0.7) = LIGHT RED
    if (corr > 0.7 && corr < 1.0) return 'rgb(255, 150, 150)';  // Light red
    
    // Perfect correlation (1.0) = DARK BLUE (diagonal)
    if (corr === 1.0) return 'rgb(10, 50, 120)';  // Very dark blue for diagonal
    
    // Blue gradient from dark to light (solid colors, no transparency)
    if (corr >= 0.6) return 'rgb(30, 90, 180)';     // Dark blue
    if (corr >= 0.5) return 'rgb(40, 110, 200)';    // Medium-dark blue
    if (corr >= 0.4) return 'rgb(60, 140, 220)';    // Medium blue
    if (corr >= 0.3) return 'rgb(80, 160, 235)';    // Medium-light blue
    if (corr >= 0.2) return 'rgb(110, 180, 245)';   // Light blue
    if (corr >= 0.1) return 'rgb(140, 200, 250)';   // Very light blue
    if (corr >= 0) return 'rgb(170, 215, 252)';     // Pale blue
    if (corr >= -0.2) return 'rgb(190, 225, 253)';  // Very pale blue
    if (corr >= -0.4) return 'rgb(210, 235, 254)';  // Almost white blue
    return 'rgb(230, 245, 255)';                     // Lightest blue
}

function renderHighCorrelationPairs(pairs) {
    const container = document.getElementById('high-corr-pairs');
    
    if (pairs.length === 0) {
        container.innerHTML = '<p style="text-align: center; color: #3fb950; padding: 20px;">✅ No pairs with high positive correlation (>0.7)</p>';
        return;
    }
    
    let html = '<div style="display: flex; flex-direction: column; gap: 8px;">';
    
    pairs.forEach(pair => {
        const num1 = pair.strat1.split('_')[0];  // Solo el número
        const num2 = pair.strat2.split('_')[0];  // Solo el número
        
        html += `
            <div class="corr-pair-item">
                <span class="corr-pair-strategies">${num1} + ${num2}</span>
                <span class="corr-pair-value">${pair.correlation.toFixed(3)}</span>
            </div>
        `;
    });
    
    html += '</div>';
    container.innerHTML = html;
}

// =============================================================================
// END CORRELATION ANALYSIS
// =============================================================================

// =============================================================================
// RISK CONTROL TAB
// =============================================================================

async function loadRiskTab() {
    try {
        // Fetch current exposure
        const exposureRes = await fetch('/api/risk/exposure');
        const exposureData = await exposureRes.json();
        
        if (!exposureData.success) {
            console.error('Failed to load risk exposure:', exposureData.error);
            return;
        }
        
        // Update global limits
        if (exposureData.limits) {
            MAX_GROSS_EXPOSURE = exposureData.limits.max_gross;
            MAX_NET_EXPOSURE = exposureData.limits.max_net;
        }
        
        const metrics = exposureData.metrics;
        const strategies = exposureData.strategies;
        
        // Update cards with dynamic colors
        updateRiskCards(metrics);
        
        // Render strategy table
        renderRiskStrategyTable(strategies, metrics.available_capital);
        
        // Load and render history chart
        await loadRiskHistoryChart();
        
    } catch (error) {
        console.error('Error loading risk tab:', error);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// QUALITY CONTROL - BINOMIAL DRIFT DETECTION
// ═══════════════════════════════════════════════════════════════════════════

async function loadBinomialDrift() {
    try {
        const response = await fetch('/api/quality/drift-binomial');
        const result = await response.json();
        
        if (!result.success) {
            console.error('Error loading binomial drift:', result.error);
            return;
        }
        
        renderBinomialDriftTable(result.data, result.window_size);
        
    } catch (error) {
        console.error('Error fetching binomial drift:', error);
    }
}

function renderBinomialDriftTable(data, windowSize) {
    const container = document.getElementById('binomial-drift-table-container');
    if (!container) {
        console.error('Binomial drift table container not found');
        return;
    }
    
    const strategies = Object.keys(data).sort();
    
    if (strategies.length === 0) {
        container.innerHTML = '<p style="text-align: center; color: #8b949e;">No data available</p>';
        return;
    }
    
    let html = `
        <table class="data-table">
            <thead>
                <tr>
                    <th>Strategy</th>
                    <th>Status</th>
                    <th>Trades</th>
                    <th>WR 100</th>
                    <th>WR 100L30</th>
                    <th>P Target</th>
                    <th>Warning Limit</th>
                    <th>Danger Limit</th>
                    <th>σ</th>
                    <th>Z-Score</th>
                    <th>Z-ScoreL30</th>
                </tr>
            </thead>
            <tbody>
    `;
    
    strategies.forEach(strategyId => {
        const item = data[strategyId];
        
        // Status badge color
        let statusColor = COLORS.healthy; // Blue for HEALTHY
        if (item.status === 'WARNING') statusColor = '#f0883e';
        if (item.status === 'DANGER') statusColor = '#f85149';
        if (item.status === 'INSUFFICIENT_DATA') statusColor = '#6e7681';
        
        // Z-score colors
        function getZScoreColor(zScore) {
            if (zScore === null || zScore === undefined) return '#8b949e';
            if (zScore < -3) return '#f85149';
            if (zScore < -2) return '#f0883e';
            if (zScore >= 0) return '#3fb950';
            return '#8b949e';
        }
        
        const zScoreColor = getZScoreColor(item.z_score);
        const zScoreL30Color = getZScoreColor(item.z_score_l30);
        
        html += `
            <tr>
                <td style="text-align: left; font-weight: 500;">${strategyId}</td>
                <td><span style="color: ${statusColor}; font-weight: 600;">${item.status}</span></td>
                <td>${item.trades_count}</td>
                <td>${item.winrate_current !== null && item.winrate_current !== undefined ? item.winrate_current.toFixed(2) + '%' : '-'}</td>
                <td>${item.winrate_l30 !== null && item.winrate_l30 !== undefined ? item.winrate_l30.toFixed(2) + '%' : '-'}</td>
                <td>${item.p_target !== null && item.p_target !== undefined ? item.p_target.toFixed(2) + '%' : '-'}</td>
                <td>${item.limit_warning !== null && item.limit_warning !== undefined ? item.limit_warning.toFixed(2) + '%' : '-'}</td>
                <td>${item.limit_danger !== null && item.limit_danger !== undefined ? item.limit_danger.toFixed(2) + '%' : '-'}</td>
                <td>${item.sigma !== null && item.sigma !== undefined ? item.sigma.toFixed(2) + '%' : '-'}</td>
                <td style="color: ${zScoreColor}; font-weight: 500;">${item.z_score !== null && item.z_score !== undefined ? item.z_score.toFixed(2) : '-'}</td>
                <td style="color: ${zScoreL30Color}; font-weight: 500;">${item.z_score_l30 !== null && item.z_score_l30 !== undefined ? item.z_score_l30.toFixed(2) : '-'}</td>
            </tr>
        `;
    });
    
    html += `
            </tbody>
        </table>
        
        <!-- Legend -->
        <div style="margin-top: 15px; padding: 12px; background: rgba(139, 148, 158, 0.1); border-radius: 6px; border-left: 3px solid #8b949e;">
            <div style="font-size: 12px; color: #8b949e; margin-bottom: 8px; font-weight: 600;">LEGEND</div>
            <div style="font-size: 12px; color: #c9d1d9; line-height: 1.8;">
                <strong>WR L30:</strong> WinRate from lagged window (shifted 30 trades back) for double confirmation<br>
                <strong>STATUS:</strong><br>
                - HEALTHY: WR Current >= Warning Limit (-2σ)<br>
                - WARNING: WR Current < Warning Limit but not confirmed<br>
                - DANGER: Both WR Current AND WR L30 < Danger Limit (-3σ, 0.13% probability each)<br>
                <strong>Z-SCORE:</strong> Standard deviations from P_target. Negative values indicate underperformance.
            </div>
        </div>
    `;
    
    container.innerHTML = html;
}
// =============================================================================
// QUALITY CONTROL TAB
// =============================================================================
async function loadQualityTab() {
    // Load binomial drift first
    await loadBinomialDrift();
    
    try {
        // Load drift analysis
        const driftRes = await fetch('/api/quality/drift');
        const driftData = await driftRes.json();
        
        if (driftData.success) {
            renderDriftTable(driftData.data);
        } else {
            document.getElementById('drift-table-container').innerHTML = 
                '<div style="text-align: center; color: #f85149; padding: 40px;">' + 
                (driftData.error || 'Error loading drift data') + 
                '</div>';
        }
        
        // Load execution quality
        const execRes = await fetch('/api/quality/execution');
        const execData = await execRes.json();
        
        if (execData.success) {
            renderExecutionTable(execData.data);
        } else {
            document.getElementById('execution-table-container').innerHTML = 
                '<div style="text-align: center; color: #f85149; padding: 40px;">' + 
                (execData.error || 'Error loading execution data') + 
                '</div>';
        }
        
        // Load target deviation
        const deviationRes = await fetch('/api/quality/target-deviation');
        const deviationData = await deviationRes.json();
        
        if (deviationData.success) {
            renderTargetDeviationTable(deviationData.data);
        } else {
            document.getElementById('deviation-table-container').innerHTML = 
                '<div style="text-align: center; color: #f85149; padding: 40px;">' + 
                (deviationData.error || 'Error loading deviation data') + 
                '</div>';
        }
        // Initialize win rate evolution checkboxes
        await initStrategyCheckboxes(
            'winrate-strategy-checkboxes',
            'winrate-strat-',
            'winrate-strat-all'
        );
        
    } catch (error) {
        console.error('Error loading quality tab:', error);
        document.getElementById('drift-table-container').innerHTML = 
            '<div style="text-align: center; color: #f85149; padding: 40px;">Error loading data</div>';
        document.getElementById('execution-table-container').innerHTML = 
            '<div style="text-align: center; color: #f85149; padding: 40px;">Error loading data</div>';
        document.getElementById('deviation-table-container').innerHTML = 
            '<div style="text-align: center; color: #f85149; padding: 40px;">Error loading data</div>';
    }
}


function renderDriftTable(data) {
    const container = document.getElementById('drift-table-container');
    
    if (!data || Object.keys(data).length === 0) {
        container.innerHTML = '<div style="text-align: center; color: #8b949e; padding: 40px;">No data available</div>';
        return;
    }
    
    // Sort by strategy ID
    const sortedStrategies = Object.keys(data).sort();
    
    let html = '<table><thead><tr>' +
            '<th>#</th>' +
            '<th>Strategy</th>' +
            '<th>Status</th>' +
            '<th>P5_Ref</th>' +
            '<th>P50_Ref</th>' +
            '<th>WinRate_100</th>' +
            '<th>WinRate_100_L30</th>' +
            '<th>Avg_Profit_100</th>' +
            '<th>Counter</th>' +
            '<th>Total Trades</th>' +
            '</tr></thead><tbody>';
    
    sortedStrategies.forEach((strategyId, index) => {
        const strat = data[strategyId];
        const num = String(index + 1).padStart(2, '0');
        
        // Status color
        let statusColor = '#8b949e';
        let statusText = strat.status;
        
        if (strat.status === 'HEALTHY') {
            statusColor = COLORS.healthy;
        } else if (strat.status === 'WARNING') {
            statusColor = COLORS.warning;
        } else if (strat.status === 'DANGER') {
            statusColor = COLORS.danger;
        }
        
        // Avg profit color
        const avgProfitColor = strat.avg_profit_100 >= 0 ? '#3fb950' : COLORS.danger;
        const avgProfitPrefix = strat.avg_profit_100 >= 0 ? '+$' : '$';
        
        // Counter color (red if > 0)
        const counterColor = strat.counter > 0 ? COLORS.danger : '#c9d1d9';
        
        html += '<tr>' +
            '<td style="color: #8b949e; font-weight: 600;">' + num + '</td>' +
            '<td>' + strategyId + '</td>' +
            '<td style="color: ' + statusColor + '; font-weight: 700; text-transform: uppercase;">' + statusText + '</td>' +
            '<td>' + (strat.p5_reference !== null ? strat.p5_reference.toFixed(1) + '%' : '-') + '</td>' +
            '<td>' + (strat.p50_reference !== null ? strat.p50_reference.toFixed(1) + '%' : '-') + '</td>' +
            '<td>' + (strat.winrate_100 !== null ? strat.winrate_100.toFixed(1) + '%' : '-') + '</td>' +
            '<td>' + (strat.winrate_100_l20 !== null ? strat.winrate_100_l20.toFixed(1) + '%' : '-') + '</td>' +
            '<td style="color: ' + avgProfitColor + ';">' + 
                (strat.avg_profit_100 !== null ? avgProfitPrefix + strat.avg_profit_100.toFixed(2) : '-') + 
            '</td>' +
            '<td style="color: ' + counterColor + '; font-weight: 600;">' + strat.counter + '</td>' +
            '<td>' + strat.total_trades + '</td>' +
            '</tr>';
    });
    
    html += '</tbody></table>';
    container.innerHTML = html;
}
function renderExecutionTable(data) {
    const container = document.getElementById('execution-table-container');
    
    if (!data || Object.keys(data).length === 0) {
        container.innerHTML = '<div style="text-align: center; color: #8b949e; padding: 40px;">No data available</div>';
        return;
    }
    
    // Sort by strategy ID
    const sortedStrategies = Object.keys(data).sort();
    
    let html = '<table><thead><tr>' +
        '<th>#</th>' +
        '<th>Strategy</th>' +
        '<th>Close Slip</th>' +
        '<th>Status</th>' +
        '<th>TP Slip</th>' +
        '<th>Status</th>' +
        '<th>SL Slip</th>' +
        '<th>Status</th>' +
        '<th>Avg Latency</th>' +
        '<th>Status</th>' +
        '<th>Total Trades</th>' +
        '</tr></thead><tbody>';
    
    sortedStrategies.forEach((strategyId, index) => {
        const strat = data[strategyId];
        const num = String(index + 1).padStart(2, '0');
        
        // Helper function for status color
        function getStatusColor(status) {
            if (status === 'HEALTHY') return COLORS.healthy;
            if (status === 'WARNING') return COLORS.warning;
            if (status === 'DANGER') return COLORS.danger;
            return '#8b949e';
        }
        
        // Format slippage values
        function formatSlippage(value) {
            if (value === null || value === undefined) return '-';
            return (value >= 0 ? '+' : '') + value.toFixed(2) + '%';
        }
        
        const closeSlippageText = formatSlippage(strat.avg_close_slippage_pct);
        const tpSlippageText = formatSlippage(strat.avg_tp_slippage_pct);
        const slSlippageText = formatSlippage(strat.avg_sl_slippage_pct);
        
        const latencyText = strat.avg_latency_sec !== null ? 
            strat.avg_latency_sec.toFixed(3) + 's' : '-';
        
        html += '<tr>' +
            '<td style="color: #8b949e; font-weight: 600;">' + num + '</td>' +
            '<td>' + strategyId + '</td>' +
            '<td>' + closeSlippageText + '</td>' +
            '<td style="color: ' + getStatusColor(strat.close_slippage_status) + '; font-weight: 700; text-transform: uppercase;">' + strat.close_slippage_status + '</td>' +
            '<td>' + tpSlippageText + '</td>' +
            '<td style="color: ' + getStatusColor(strat.tp_slippage_status) + '; font-weight: 700; text-transform: uppercase;">' + strat.tp_slippage_status + '</td>' +
            '<td>' + slSlippageText + '</td>' +
            '<td style="color: ' + getStatusColor(strat.sl_slippage_status) + '; font-weight: 700; text-transform: uppercase;">' + strat.sl_slippage_status + '</td>' +
            '<td>' + latencyText + '</td>' +
            '<td style="color: ' + getStatusColor(strat.latency_status) + '; font-weight: 700; text-transform: uppercase;">' + strat.latency_status + '</td>' +
            '<td>' + strat.total_trades + '</td>' +
            '</tr>';
    });
    
    html += '</tbody></table>';
    container.innerHTML = html;
}
function renderTargetDeviationTable(data) {
    const container = document.getElementById('deviation-table-container');
    
    if (!data || Object.keys(data).length === 0) {
        container.innerHTML = '<div style="text-align: center; color: #8b949e; padding: 40px;">No data available</div>';
        return;
    }
    
    // Sort by strategy ID
    const sortedStrategies = Object.keys(data).sort();
    
    let html = '<table><thead><tr>' +
        '<th>#</th>' +
        '<th>Strategy</th>' +
        '<th>TP Real%</th>' +
        '<th>TP Target%</th>' +
        '<th>TP Dev</th>' +
        '<th>SL Real%</th>' +
        '<th>SL Target%</th>' +
        '<th>SL Dev</th>' +
        '<th>Total Trades</th>' +
        '</tr></thead><tbody>';
    
    sortedStrategies.forEach((strategyId, index) => {
        const strat = data[strategyId];
        const num = String(index + 1).padStart(2, '0');
        
        // Helper function for deviation color
        function getDeviationColor(deviation) {
            if (deviation === null || deviation === undefined) return '#8b949e';
            
            // Positive deviation is ALWAYS good (we got more profit than expected)
            if (deviation > 0) return '#3fb950';  // 🟢 Green
            
            // Negative deviation: apply thresholds
            const absDev = Math.abs(deviation);
            if (absDev < 0.2) return '#3fb950';  
            if (absDev < 0.5) return COLORS.warning;  
            return COLORS.danger;                   
        }
        
        // Format percentage values
        function formatPct(value) {
            if (value === null || value === undefined) return '-';
            return (value >= 0 ? '+' : '') + value.toFixed(2) + '%';
        }
        
        const tpRealText = formatPct(strat.tp_real_pct);
        const tpTargetText = formatPct(strat.tp_target_pct);
        const tpDevText = formatPct(strat.tp_deviation);
        
        const slRealText = formatPct(strat.sl_real_pct);
        const slTargetText = formatPct(strat.sl_target_pct);
        const slDevText = formatPct(strat.sl_deviation);
        
        const tpDevColor = getDeviationColor(strat.tp_deviation);
        const slDevColor = getDeviationColor(strat.sl_deviation);
        
        // Total trades = TP trades + SL trades
        const totalTrades = strat.tp_trades + strat.sl_trades;
        
        html += '<tr>' +
            '<td style="color: #8b949e; font-weight: 600;">' + num + '</td>' +
            '<td>' + strategyId + '</td>' +
            '<td>' + tpRealText + '</td>' +
            '<td>' + tpTargetText + '</td>' +
            '<td style="color: ' + tpDevColor + '; font-weight: 600;">' + tpDevText + '</td>' +
            '<td>' + slRealText + '</td>' +
            '<td>' + slTargetText + '</td>' +
            '<td style="color: ' + slDevColor + '; font-weight: 600;">' + slDevText + '</td>' +
            '<td>' + totalTrades + '</td>' +
            '</tr>';
    });
    
    html += '</tbody></table>';
    container.innerHTML = html;
}
// =============================================================================
// END QUALITY CONTROL TAB
// =============================================================================

function updateRiskCards(metrics) {
    const grossPct = metrics.gross_exposure_pct;
    const netPct = metrics.net_exposure_pct;
    const longPct = metrics.long_exposure_pct;
    const shortPct = metrics.short_exposure_pct;
    
    // Update Gross Exposure card (always blue)
    const grossEl = document.getElementById('risk-gross-exp');
    if (grossEl) {
        grossEl.textContent = grossPct.toFixed(1) + '%';
        grossEl.style.color = '#58a6ff';  // Blue
    }
    
    // Update Net Exposure card (always blue)
    const netEl = document.getElementById('risk-net-exp');
    if (netEl) {
        const netSign = netPct >= 0 ? '+' : '';
        netEl.textContent = netSign + netPct.toFixed(1) + '%';
        netEl.style.color = '#58a6ff';  // Blue
    }
    
    // Update Long Exposure card (always green)
    const longEl = document.getElementById('risk-long-exp');
    if (longEl) {
        longEl.textContent = longPct.toFixed(1) + '%';
    }
    
    // Update Short Exposure card (always red)
    const shortEl = document.getElementById('risk-short-exp');
    if (shortEl) {
        shortEl.textContent = shortPct.toFixed(1) + '%';
    }
    
    // Update config card
    const configGrossEl = document.getElementById('risk-config-max-gross');
    const configNetEl = document.getElementById('risk-config-max-net');
    
    if (configGrossEl) {
        configGrossEl.textContent = MAX_GROSS_EXPOSURE.toFixed(1) + '%';
    }
    
    if (configNetEl) {
        configNetEl.textContent = MAX_NET_EXPOSURE.toFixed(1) + '%';
    }
}

function renderRiskStrategyTable(strategies, availableCapital) {
    const container = document.getElementById('risk-strategy-table');
    
    if (!strategies || strategies.length === 0) {
        container.innerHTML = '<div style="text-align: center; color: #8b949e; padding: 40px;">No open positions</div>';
        return;
    }
    
    // Calculate total gross exposure (sum of all strategy exposures)
    const totalGrossExposure = strategies.reduce((sum, strat) => sum + strat.pct, 0);
    
    const html = '<table><thead><tr>' +
        '<th>#</th>' +
        '<th>Strategy</th>' +
        '<th>Side</th>' +
        '<th>USDT</th>' +
        '<th>% Exposure</th>' +
        '<th>% of Total</th>' +
        '</tr></thead><tbody>' +
        strategies.map((strat, idx) => {
            const num = String(idx + 1).padStart(2, '0');
            const sideClass = strat.side === 'LONG' ? 'direction-long' : 'direction-short';
            
            // Calculate % of total gross exposure
            const pctOfTotal = totalGrossExposure > 0 
                ? (strat.pct / totalGrossExposure * 100) 
                : 0;
            
            return '<tr>' +
                '<td style="color: #8b949e; font-weight: 600;">' + num + '</td>' +
                '<td>' + strat.strategy + '</td>' +
                '<td class="' + sideClass + '">' + strat.side + '</td>' +
                '<td>$' + strat.usdt.toFixed(2) + '</td>' +
                '<td>' + strat.pct.toFixed(2) + '%</td>' +
                '<td>' + pctOfTotal.toFixed(1) + '%</td>' +
                '</tr>';
        }).join('') +
        '</tbody></table>';
    
    container.innerHTML = html;
}

async function loadRiskHistoryChart() {
    try {
        const dateParams = getRiskDateParams();
        const res = await fetch('/api/risk/exposure-history?days=30' + dateParams);
        const data = await res.json();
        
        if (!data.success || !data.history || data.history.dates.length === 0) {
            console.log('No risk history data available yet');
            return;
        }
        
        const history = data.history;
        
        // Destroy existing chart
        if (riskExposureChart) {
            riskExposureChart.destroy();
            riskExposureChart = null;
        }
        
        // Fetch BTC data for overlay
        let btcPrices = [];
        try {
            const btcRes = await fetch('/api/btc/history' + dateParams);
            const btcData = await btcRes.json();
            
            if (btcData.success && btcData.dates && btcData.dates.length > 0) {
                // Create map: date -> price
                const btcMap = {};
                btcData.dates.forEach((date, idx) => {
                    btcMap[date] = btcData.prices[idx];
                });
                
                // Align with exposure dates
                btcPrices = history.dates.map(date => btcMap[date] || null);
            }
        } catch (error) {
            console.error('Error loading BTC data for risk chart:', error);
        }
        
        // Calculate Y2 axis range for BTC (min/max with 5% padding)
        let btcMin = Math.min(...btcPrices.filter(p => p !== null));
        let btcMax = Math.max(...btcPrices.filter(p => p !== null));
        const btcPadding = (btcMax - btcMin) * 0.05;
        btcMin -= btcPadding;
        btcMax += btcPadding;
        
        // Gross Exposure always blue
        const grossColor = '#58a6ff';  // Blue
        
        const ctx = document.getElementById('riskExposureChart').getContext('2d');
        riskExposureChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: history.dates,
                datasets: [
                    {
                        label: 'Gross Exposure (%)',
                        data: history.gross,
                        borderColor: grossColor,
                        backgroundColor: 'transparent',
                        borderWidth: 2,
                        pointRadius: 3,
                        pointBackgroundColor: grossColor,
                        tension: 0.1,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Net Exposure (%)',
                        data: history.net,
                        borderColor: '#22d3ee',
                        backgroundColor: 'transparent',
                        borderWidth: 1.5,
                        pointRadius: 2,
                        pointBackgroundColor: '#22d3ee',
                        tension: 0.1,
                        yAxisID: 'y'
                    },
                    {
                        label: 'BTC Price',
                        data: btcPrices,
                        borderColor: '#f59e0b',
                        backgroundColor: 'transparent',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        pointRadius: 0,
                        tension: 0.1,
                        yAxisID: 'y2'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            color: '#ffffff',
                            font: { size: 14 }
                        }
                    },
                    title: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            color: '#ffffff',
                            font: { size: 14 }
                        },
                        grid: {
                            color: '#21262d',
                            drawBorder: true,
                            borderColor: '#facc15',
                            borderWidth: 1
                        }
                    },
                    y: {
                        type: 'linear',
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Exposure (%)',
                            color: '#ffffff',
                            font: { size: 14 }
                        },
                        ticks: {
                            color: '#ffffff',
                            font: { size: 14 },
                            callback: function(value) {
                                return value.toFixed(1) + '%';
                            }
                        },
                        grid: {
                            color: '#21262d',
                            drawBorder: true,
                            borderColor: '#facc15',
                            borderWidth: 1
                        }
                    },
                    y2: {
                        type: 'linear',
                        position: 'right',
                        min: btcMin,
                        max: btcMax,
                        title: {
                            display: true,
                            text: 'BTC Price ($)',
                            color: '#f59e0b',
                            font: { size: 14 }
                        },
                        ticks: {
                            color: '#f59e0b',
                            font: { size: 14 },
                            callback: function(value) {
                                return '$' + value.toLocaleString(undefined, {
                                    minimumFractionDigits: 0,
                                    maximumFractionDigits: 0
                                });
                            }
                        },
                        grid: {
                            drawOnChartArea: false
                        }
                    }
                }
            }
        });
        
    } catch (error) {
        console.error('Error loading risk history chart:', error);
    }
}

// =============================================================================
// WIN RATE EVOLUTION CHART (Quality Control)
// =============================================================================

let winRateChart = null;

function clearWinRateDates() {
    document.getElementById('winrate-date-from').value = '';
    document.getElementById('winrate-date-to').value = '';
}

function getWinRateDateParams() {
    const dateFrom = document.getElementById('winrate-date-from').value;
    const dateTo = document.getElementById('winrate-date-to').value;
    let params = '';
    if (dateFrom) params += '&date_from=' + dateFrom;
    if (dateTo) params += '&date_to=' + dateTo;
    return params;
}

async function updateWinRateChart() {
    try {
        const selectedStrategies = getSelectedStrategies('winrate-strategy-checkboxes');
        
        if (selectedStrategies.length === 0) {
            alert('Please select at least one strategy');
            return;
        }
        
        const dateParams = getWinRateDateParams();
        const res = await fetch('/api/quality/winrate-evolution?strategies=' + selectedStrategies.join(',') + dateParams);
        const data = await res.json();
        
        if (!data.success) {
            alert('Error loading win rate data: ' + (data.error || 'Unknown error'));
            return;
        }
        
        if (!data.dates || data.dates.length === 0) {
            alert('No trades found for selected strategies in this date range');
            return;
        }
        
        // Destroy existing chart
        if (winRateChart) {
            winRateChart.destroy();
            winRateChart = null;
        }
        
        // Create chart
        const ctx = document.getElementById('winRateChart').getContext('2d');
        winRateChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.dates,
                datasets: [{
                    label: 'Cumulative Win Rate (%)',
                    data: data.winrate,
                    borderColor: '#58a6ff',
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    pointRadius: 2,
                    pointBackgroundColor: '#58a6ff',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            color: '#ffffff',
                            font: { size: 14 }
                        }
                    },
                    title: {
                        display: true,
                        text: 'Win Rate Evolution - ' + selectedStrategies.length + ' strategies (' + data.total_trades + ' trades)',
                        color: CHART_DEFAULTS.titleColor,
                        font: { size: CHART_DEFAULTS.fontSize.title, weight: 'bold' }
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            color: CHART_DEFAULTS.textColor,
                            font: { size: CHART_DEFAULTS.fontSize.axis }
                        },
                        grid: {
                            color: CHART_DEFAULTS.gridColor,
                            drawBorder: true,
                            borderColor: CHART_DEFAULTS.borderColor,
                            borderWidth: CHART_DEFAULTS.borderWidth
                        }
                    },
                    y: {
                        ticks: {
                            color: CHART_DEFAULTS.textColor,
                            font: { size: CHART_DEFAULTS.fontSize.axis },
                            callback: function(value) { return value.toFixed(1) + '%'; }
                        },
                        grid: {
                            color: CHART_DEFAULTS.gridColor,
                            drawBorder: true,
                            borderColor: CHART_DEFAULTS.borderColor,
                            borderWidth: CHART_DEFAULTS.borderWidth
                        }
                    }
                }
            }
        });
        
    } catch (error) {
        console.error('Error updating win rate chart:', error);
        alert('Error loading win rate chart: ' + error.message);
    }
}

// =============================================================================
// END WIN RATE EVOLUTION CHART
// =============================================================================
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeDashboard);
} else {
    initializeDashboard();
}

window.addEventListener('pageshow', function(event) {
    if (event.persisted) {
        initializeDashboard();
    }
});