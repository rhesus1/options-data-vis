const correctPassword = "F0ur13rM0rg4n%";
let isAuthenticated = false;
let currentData = {
    ranking: [],
    stock: [],
    summary: [],
    topVolume: [],
    topOpenInterest: [],
    historic: [],
    processed: []
};

function checkPassword() {
    const password = prompt("Please enter the password:");
    if (password === correctPassword) {
        isAuthenticated = true;
        document.body.style.display = "flex";
    } else {
        alert("Incorrect password!");
        document.body.innerHTML = "<h1>Access Denied</h1>";
    }
}

function interpolateGrid(points, values, gridX, gridY) {
    // Bilinear interpolation for 2D grid
    const result = [];
    for (let i = 0; i < gridY.length; i++) {
        const row = [];
        for (let j = 0; j < gridX.length; j++) {
            // Find four nearest points for bilinear interpolation
            let closest = points
                .map((p, idx) => ({ x: p[0], y: p[1], value: values[idx] }))
                .filter(p => p.x !== null && p.y !== null && !isNaN(p.value))
                .sort((a, b) => {
                    const distA = Math.sqrt((a.x - gridX[j]) ** 2 + (a.y - gridY[i]) ** 2);
                    const distB = Math.sqrt((b.x - gridX[j]) ** 2 + (b.y - gridY[i]) ** 2);
                    return distA - distB;
                })
                .slice(0, 4);
            
            if (closest.length < 4) {
                // If fewer than 4 points, use linear interpolation or nearest neighbor
                if (closest.length === 0) {
                    row.push(0); // Fallback to zero if no points
                    continue;
                }
                row.push(closest[0].value); // Nearest neighbor
                continue;
            }

            // Bilinear interpolation
            const [p1, p2, p3, p4] = closest;
            const x1 = Math.min(p1.x, p2.x), x2 = Math.max(p1.x, p2.x);
            const y1 = Math.min(p1.y, p3.y), y2 = Math.max(p1.y, p3.y);
            
            if (x1 === x2 || y1 === y2) {
                row.push(p1.value); // Fallback to nearest if points are aligned
                continue;
            }

            const f11 = closest.find(p => p.x === x1 && p.y === y1)?.value || p1.value;
            const f12 = closest.find(p => p.x === x1 && p.y === y2)?.value || p1.value;
            const f21 = closest.find(p => p.x === x2 && p.y === y1)?.value || p1.value;
            const f22 = closest.find(p => p.x === x2 && p.y === y2)?.value || p1.value;

            const fx1 = f11 + (gridX[j] - x1) * (f21 - f11) / (x2 - x1);
            const fx2 = f12 + (gridX[j] - x1) * (f22 - f12) / (x2 - x1);
            const value = fx1 + (gridY[i] - y1) * (fx2 - fx1) / (y2 - y1);
            row.push(isNaN(value) ? p1.value : value); // Ensure no NaN
        }
        result.push(row);
    }
    return result;
}


function updateCallVolSurface() {
    const ticker = document.getElementById('ticker-search').value.toUpperCase();
    const timestamp = document.getElementById('date-select').value + '_' + document.getElementById('time-select').value;
    const source = document.getElementById('source-select').value;
    const prefix = source === 'yfinance' ? '_yfinance' : '';
    const moneynessRange = document.getElementById('moneyness-slider').noUiSlider.get();
    const expiryTRange = document.getElementById('expiry-t-slider').noUiSlider.get();
    const moneynessMin = parseFloat(moneynessRange[0]);
    const moneynessMax = parseFloat(moneynessRange[1]);
    const expiryTMin = parseFloat(expiryTRange[0]);
    const expiryTMax = parseFloat(expiryTRange[1]);

    const processedData = currentData.processed.filter(row => row.Ticker === ticker && row.Type === 'Call' && row.Moneyness && row.Smoothed_IV && !isNaN(row.Moneyness) && !isNaN(row.Smoothed_IV));
    if (processedData.length === 0) {
        document.getElementById('call-vol-surface').innerHTML = '';
        document.getElementById('call-vol-surface-error').textContent = 'No data available';
        document.getElementById('call-vol-surface-error').style.display = 'block';
        return;
    }

    const datasetDate = new Date(parseInt(timestamp.slice(0, 4)), parseInt(timestamp.slice(4, 6)) - 1, parseInt(timestamp.slice(6, 8)));
    processedData.forEach(row => {
        if (row.Expiry) {
            const expiryDate = new Date(row.Expiry);
            row.Expiry_T = (expiryDate - datasetDate) / (1000 * 60 * 60 * 24 * 365);
        }
    });

    const filteredData = processedData.filter(row => 
        row.Moneyness >= moneynessMin && row.Moneyness <= moneynessMax &&
        row.Expiry_T >= expiryTMin && row.Expiry_T <= expiryTMax && row.Expiry_T !== null
    );

    if (filteredData.length === 0) {
        document.getElementById('call-vol-surface').innerHTML = '';
        document.getElementById('call-vol-surface-error').textContent = 'No data available';
        document.getElementById('call-vol-surface-error').style.display = 'block';
        return;
    }

    const moneynessValues = Array.from({length: Math.floor((moneynessMax - moneynessMin) / 0.05) + 1}, (_, i) => moneynessMin + i * 0.05);
    const expiryValues = [...new Set(filteredData.map(row => row.Expiry).filter(e => e))].sort();
    const expiryTimes = [...new Set(filteredData.map(row => row.Expiry_T).filter(t => t !== null))].sort().slice(0, 50); // Cap at 50 expiries

    if (expiryTimes.length === 0 || moneynessValues.length === 0) {
        document.getElementById('call-vol-surface').innerHTML = '';
        document.getElementById('call-vol-surface-error').textContent = 'No valid expiries/moneyness';
        document.getElementById('call-vol-surface-error').style.display = 'block';
        return;
    }

    const points = filteredData.map(row => [row.Expiry_T, row.Moneyness]);
    const values = filteredData.map(row => row.Smoothed_IV * 100);

    const z = interpolateGrid(points, values, expiryTimes, moneynessValues);

    const data = [{
        x: expiryTimes,
        y: moneynessValues,
        z: z,
        type: 'surface',
        colorscale: 'Viridis',
        showscale: true
    }];

    const layout = {
        title: 'Call Volatility Surface',
        scene: {
            xaxis: {title: 'Expiry (T)'},
            yaxis: {title: 'Moneyness'},
            zaxis: {title: 'Volatility (%)'}
        },
        margin: {l: 40, r: 40, t: 40, b: 40}
    };

    document.getElementById('call-vol-surface-error').style.display = 'none';
    Plotly.newPlot('call-vol-surface', data, layout);
}

async function loadData(timestamp) {
    const source = document.getElementById('source-select').value;
    const ticker = document.getElementById('ticker-search').value.toUpperCase();
    const prefix = source === 'yfinance' ? '_yfinance' : '';
    try {
        const files = [
            {url: `data/${timestamp}/tables/ranking/ranking_table${prefix}.csv?v=${Date.now()}`, key: 'ranking'},
            {url: `data/${timestamp}/tables/stock/stock_table${prefix}.csv?v=${Date.now()}`, key: 'stock'},
            {url: `data/${timestamp}/tables/summary/summary_table${prefix}.csv?v=${Date.now()}`, key: 'summary'},
            {url: `data/${timestamp}/tables/contracts/top_volume_table${prefix}.csv?v=${Date.now()}`, key: 'topVolume'},
            {url: `data/${timestamp}/tables/contracts/top_open_interest_table${prefix}.csv?v=${Date.now()}`, key: 'topOpenInterest'},
            {url: `data/${timestamp}/processed${prefix}/processed${prefix}_${ticker}.csv?v=${Date.now()}`, key: 'processed'},
            {url: `data/${timestamp}/historic/historic_${ticker}.csv?v=${Date.now()}`, key: 'historic'}
        ];

        const promises = files.map(file => 
            fetch(file.url)
                .then(response => response.ok ? response.text() : Promise.reject(`Failed to load ${file.url}`))
                .then(text => new Promise((resolve, reject) => {
                    Papa.parse(text, {
                        header: true,
                        skipEmptyLines: true,
                        complete: (result) => resolve({key: file.key, data: result.data}),
                        error: (error) => reject(error)
                    });
                }))
                .catch(error => {
                    console.warn(`Error loading ${file.url}: ${error}`);
                    return {key: file.key, data: []};
                })
        );

        const results = await Promise.all(promises);
        results.forEach(result => {
            currentData[result.key] = result.data;
        });

        console.log('Data loaded for timestamp:', timestamp);
    } catch (error) {
        console.error('Error loading data:', error);
        document.getElementById('historic-error').textContent = `Error loading data: ${error.message}`;
        document.getElementById('historic-error').style.display = 'block';
    }
}

function updateSummaryTable() {
    try {
        const ticker = document.getElementById('ticker-search').value.toUpperCase();
        const table = document.getElementById('summary-table');
        table.innerHTML = '';

        const filteredData = currentData.summary.filter(row => row.Ticker === ticker);
        if (filteredData.length === 0) {
            table.innerHTML = '<tr><td colspan="2">No summary data available</td></tr>';
            return;
        }

        const row = filteredData[0];
        const thead = document.createElement('thead');
        const headerRow = document.createElement('tr');
        ['Metric', 'Value'].forEach(col => {
            const th = document.createElement('th');
            th.textContent = col;
            headerRow.appendChild(th);
        });
        thead.appendChild(headerRow);
        table.appendChild(thead);

        const tbody = document.createElement('tbody');
        const metrics = [
            "Latest Close ($)", "Open ($)", "Low ($)", "High ($)", "Daily Volume",
            "Close 1d (%)", "Close 1w (%)", "Realised Volatility 100d (%)",
            "Weighted IV (%)", "Weighted IV 1d (%)", "Weighted IV 1w (%)",
            "Volume 1d (%)", "Volume 1w (%)", "Open Interest", "OI 1d (%)",
            "OI 1w (%)", "ATM 12m/3m Ratio"
        ];

        metrics.forEach(metric => {
            const tr = document.createElement('tr');
            const metricCell = document.createElement('td');
            metricCell.textContent = metric;
            tr.appendChild(metricCell);

            const valueCell = document.createElement('td');
            const value = row[metric];
            valueCell.textContent = value || 'N/A';
            if (row[`${metric}_Color`]) {
                valueCell.style.backgroundColor = row[`${metric}_Color`];
            }
            tr.appendChild(valueCell);
            tbody.appendChild(tr);
        });

        table.appendChild(tbody);
    } catch (error) {
        console.error('Error updating summary table:', error);
        document.getElementById('summary-table').innerHTML = '<tr><td colspan="2">Error loading summary data</td></tr>';
    }
}

function updateContractsTable() {
    try {
        const ticker = document.getElementById('ticker-search').value.toUpperCase();
        const volumeTable = document.getElementById('top-volume-table');
        const openInterestTable = document.getElementById('top-open-interest-table');
        volumeTable.innerHTML = '';
        openInterestTable.innerHTML = '';

        const tables = [
            {table: volumeTable, data: currentData.topVolume.filter(row => row.Ticker === ticker), id: 'top-volume-table'},
            {table: openInterestTable, data: currentData.topOpenInterest.filter(row => row.Ticker === ticker), id: 'top-open-interest-table'}
        ];

        tables.forEach(({table, data, id}) => {
            if (data.length === 0) {
                table.innerHTML = `<tr><td colspan="7">No ${id.includes('volume') ? 'volume' : 'open interest'} data available for ${ticker}</td></tr>`;
                return;
            }

            const thead = document.createElement('thead');
            const headerRow = document.createElement('tr');
            const columns = ['Ticker', 'Strike', 'Expiry', 'Type', 'Bid', 'Ask', 'Volume', 'Open Interest'];
            columns.forEach((col, index) => {
                const th = document.createElement('th');
                th.textContent = col;
                th.dataset.column = col;
                th.dataset.order = 'asc';
                th.addEventListener('click', () => sortTable(id, index));
                headerRow.appendChild(th);
            });
            thead.appendChild(headerRow);
            table.appendChild(thead);

            const tbody = document.createElement('tbody');
            data.forEach(item => {
                const row = tbody.insertRow();
                columns.forEach(col => {
                    const cell = row.insertCell();
                    let value = item[col];
                    if (Number.isFinite(value)) {
                        if (col === 'Volume' || col === 'Open Interest') {
                            value = parseInt(value).toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 });
                        } else {
                            value = value.toFixed(2);
                        }
                    } else {
                        value = value || 'N/A';
                    }
                    cell.textContent = value;
                });
            });
            table.appendChild(tbody);
        });
    } catch (error) {
        console.error('Error updating contracts table:', error);
        document.getElementById('top-volume-table').innerHTML = '<tr><td colspan="7">Error loading volume data</td></tr>';
        document.getElementById('top-open-interest-table').innerHTML = '<tr><td colspan="7">Error loading open interest data</td></tr>';
    }
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function updateSection(sectionId) {
    document.querySelectorAll('.overview-grid, .volatility-grid, .chart-container').forEach(el => {
        el.style.display = el.id === sectionId ? (el.classList.contains('overview-grid') || el.classList.contains('volatility-grid') ? 'grid' : 'block') : 'none';
    });
    if (sectionId === 'overview') {
        updateHistoricChart();
        updateSummaryTable();
        updateContractsTable();
    } else if (sectionId === 'volatility') {
        updateCharts();
        updateCallVolSurface();
    } else if (sectionId === 'ranking') {
        updateRankingCharts();
        updateRankingTable();
    } else if (sectionId === 'stock') {
        updateStockTable();
    } else if (sectionId === 'raw-data') {
        updateRawDataTable();
    }
}

function updateDropdowns() {
    const ticker = document.getElementById('ticker-search').value.toUpperCase();
    const expirySelect = document.getElementById('expiry-select');
    const yfExpirySelect = document.getElementById('yf-expiry-select');
    const moneynessSelect = document.getElementById('moneyness-select');

    const processedData = currentData.processed.filter(row => row.Ticker === ticker && row.Type === 'Call');
    const expiries = [...new Set(processedData.map(row => row.Expiry).filter(e => e))].sort();
    expirySelect.innerHTML = '';
    yfExpirySelect.innerHTML = '';
    expiries.forEach(expiry => {
        const option = document.createElement('option');
        option.value = expiry;
        option.textContent = expiry;
        expirySelect.appendChild(option);
        yfExpirySelect.appendChild(option.cloneNode(true));
    });

    const moneynessValues = [...new Set(processedData.map(row => parseFloat(row.Moneyness)).filter(m => !isNaN(m)))].sort((a, b) => a - b);
    moneynessSelect.innerHTML = '';
    moneynessValues.forEach(m => {
        const option = document.createElement('option');
        option.value = m;
        option.textContent = m.toFixed(2);
        moneynessSelect.appendChild(option);
    });
}

function updateHistoricChart() {
    // Implementation unchanged (loads historic_<ticker>.csv)
}

function updateCharts() {
    // Implementation unchanged (uses processed data for IV smile, term structure, etc.)
}

function updateRankingCharts() {
    // Implementation unchanged (uses ranking_table_yfinance.csv)
}

function updateRankingTable() {
    try {
        const table = document.getElementById('ranking-table');
        table.innerHTML = '';
        if (currentData.ranking.length === 0) {
            table.innerHTML = '<tr><td colspan="35">No ranking data available</td></tr>';
            return;
        }

        const thead = document.createElement('thead');
        const headerRow = document.createElement('tr');
        const columns = [
            "Rank", "Ticker", "Company Name", "Latest Close", "Realised Volatility 30d (%)",
            "Realised Volatility 100d (%)", "Realised Volatility 100d 1d (%)",
            "Realised Volatility 100d 1w (%)", "Min Realised Volatility 100d (1y)",
            "Max Realised Volatility 100d (1y)", "Mean Realised Volatility 100d (1y)",
            "Rvol 100d Percentile (%)", "Rvol 100d Z-Score Percentile (%)",
            "Realised Volatility 180d (%)", "Realised Volatility 252d (%)",
            "Weighted IV (%)", "Weighted IV 1d (%)", "Weighted IV 1w (%)",
            "Weighted IV 3m (%)", "Weighted IV 3m 1d (%)", "Weighted IV 3m 1w (%)",
            "ATM IV 3m (%)", "ATM IV 3m 1d (%)", "ATM IV 3m 1w (%)",
            "Rvol100d - Weighted IV", "Volume", "Volume 1d (%)", "Volume 1w (%)",
            "Open Interest", "OI 1d (%)", "OI 1w (%)", "Market Spread", "1y Spread",
            "3y Spread", "5y Spread"
        ];
        columns.forEach((col, index) => {
            const th = document.createElement('th');
            th.textContent = col;
            th.dataset.column = col;
            th.dataset.order = 'asc';
            th.addEventListener('click', () => sortTable('ranking-table', index));
            headerRow.appendChild(th);
        });
        thead.appendChild(headerRow);
        table.appendChild(thead);

        const tbody = document.createElement('tbody');
        currentData.ranking.forEach(item => {
            const row = tbody.insertRow();
            columns.forEach(col => {
                const cell = row.insertCell();
                let value = item[col];
                if (Number.isFinite(value)) {
                    if (col === 'Volume' || col === 'Open Interest') {
                        value = parseInt(value).toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 });
                    } else {
                        value = value.toFixed(2);
                    }
                } else {
                    value = value || 'N/A';
                }
                cell.textContent = value;
                if (item[`${col}_Color`]) {
                    cell.style.backgroundColor = item[`${col}_Color`];
                }
            });
        });
        table.appendChild(tbody);
    } catch (error) {
        console.error('Error updating ranking table:', error);
        document.getElementById('ranking-table').innerHTML = '<tr><td colspan="35">Error loading ranking data</td></tr>';
    }
}

function updateStockTable() {
    try {
        const table = document.getElementById('stock-table');
        table.innerHTML = '';
        if (currentData.stock.length === 0) {
            table.innerHTML = '<tr><td colspan="18">No stock data available</td></tr>';
            return;
        }

        const thead = document.createElement('thead');
        const headerRow = document.createElement('tr');
        const columns = [
            "Ticker", "Company Name", "Latest Open", "Latest Close", "Latest High",
            "Latest Low", "Open 1d (%)", "Open 1w (%)", "Close 1d (%)", "Close 1w (%)",
            "High 1d (%)", "High 1w (%)", "Low 1d (%)", "Low 1w (%)", "Market Spread",
            "1y Spread", "3y Spread", "5y Spread"
        ];
        columns.forEach((col, index) => {
            const th = document.createElement('th');
            th.textContent = col;
            th.dataset.column = col;
            th.dataset.order = 'asc';
            th.addEventListener('click', () => sortTable('stock-table', index));
            headerRow.appendChild(th);
        });
        thead.appendChild(headerRow);
        table.appendChild(thead);

        const tbody = document.createElement('tbody');
        currentData.stock.forEach(item => {
            const row = tbody.insertRow();
            columns.forEach(col => {
                const cell = row.insertCell();
                let value = item[col];
                if (Number.isFinite(value)) {
                    value = value.toFixed(2);
                } else {
                    value = value || 'N/A';
                }
                cell.textContent = value;
                if (item[`${col}_Color`]) {
                    cell.style.backgroundColor = item[`${col}_Color`];
                }
            });
        });
        table.appendChild(tbody);
    } catch (error) {
        console.error('Error updating stock table:', error);
        document.getElementById('stock-table').innerHTML = '<tr><td colspan="18">Error loading stock data</td></tr>';
    }
}

function updateRawDataTable() {
    try {
        const ticker = document.getElementById('ticker-search').value.toUpperCase();
        const table = document.getElementById('raw-data-table');
        table.innerHTML = '';
        const cleanedData = currentData.processed.filter(row => row.Ticker === ticker);
        if (cleanedData.length === 0) {
            table.innerHTML = '<tr><td colspan="10">No raw data available</td></tr>';
            return;
        }

        const thead = document.createElement('thead');
        const headerRow = document.createElement('tr');
        const columns = ['Ticker', 'Strike', 'Expiry', 'Type', 'Bid', 'Ask', 'Volume', 'Open Interest', 'Moneyness', 'Smoothed_IV'];
        columns.forEach((col, index) => {
            const th = document.createElement('th');
            th.textContent = col;
            th.dataset.column = col;
            th.dataset.order = 'asc';
            th.addEventListener('click', () => sortTable('raw-data-table', index));
            headerRow.appendChild(th);
        });
        thead.appendChild(headerRow);
        table.appendChild(thead);

        const tbody = document.createElement('tbody');
        cleanedData.forEach(item => {
            const row = tbody.insertRow();
            columns.forEach(col => {
                const cell = row.insertCell();
                let value = item[col];
                if (Number.isFinite(value)) {
                    if (col === 'Volume' || col === 'Open Interest') {
                        value = parseInt(value).toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 });
                    } else {
                        value = value.toFixed(2);
                    }
                } else {
                    value = value || 'N/A';
                }
                cell.textContent = value;
            });
        });
        table.appendChild(tbody);
    } catch (error) {
        console.error('Error updating raw data table:', error);
        document.getElementById('raw-data-table').innerHTML = '<tr><td colspan="10">Error loading raw data table</td></tr>';
    }
}

function sortTable(tableId, columnIndex) {
    try {
        const table = document.getElementById(tableId);
        const header = table.querySelector('thead tr').children[columnIndex];
        const column = header.dataset.column;
        const order = header.dataset.order === 'desc' ? 'asc' : 'desc';
        header.dataset.order = order;
        const tbody = table.querySelector('tbody');
        const rows = Array.from(tbody.querySelectorAll('tr'));
        rows.sort((a, b) => {
            let aValue = a.children[columnIndex].textContent;
            let bValue = b.children[columnIndex].textContent;
            if (aValue === 'N/A') return 1;
            if (bValue === 'N/A') return -1;
            if (column === 'Volume' || column === 'Open Interest') {
                aValue = parseInt(aValue.replace(/,/g, '')) || -Infinity;
                bValue = parseInt(bValue.replace(/,/g, '')) || -Infinity;
            } else if (column === 'Rank') {
                aValue = parseInt(aValue);
                bValue = parseInt(bValue);
            } else {
                aValue = parseFloat(aValue) || aValue;
                bValue = parseFloat(bValue) || bValue;
            }
            if (typeof aValue === 'number' && typeof bValue === 'number') {
                return order === 'desc' ? bValue - aValue : aValue - bValue;
            }
            return order === 'desc' ? bValue.localeCompare(aValue) : aValue.localeCompare(bValue);
        });
        tbody.innerHTML = '';
        rows.forEach(row => tbody.appendChild(row));
    } catch (error) {
        console.error('Error sorting table:', error);
    }
}

function addKeyboardNavigation() {
    try {
        const chartConfigs = [
            { canvasId: 'moneyness-vs-iv-chart', selectId: 'expiry-select' },
            { canvasId: 'expiry-vs-iv-chart', selectId: 'moneyness-select' },
            { canvasId: 'yf-iv-chart', selectId: 'yf-expiry-select' },
            { canvasId: 'skew-vs-expiry-chart', selectId: null },
            { canvasId: 'realised-vol-chart', selectId: 'realised-vol-history-select' },
            { canvasId: 'volume-chart', selectId: 'ticker-search' },
            { canvasId: 'open-interest-chart', selectId: 'ticker-search' },
            { canvasId: 'historic-price-chart', selectId: 'history-select' }
        ];
        chartConfigs.forEach(config => {
            const canvas = document.getElementById(config.canvasId);
            const select = config.selectId ? document.getElementById(config.selectId) : null;
            if (select) {
                canvas.addEventListener('keydown', (event) => {
                    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
                        event.preventDefault();
                        let newIndex = select.selectedIndex + (event.key === 'ArrowLeft' ? -1 : 1);
                        if (newIndex < 0) newIndex = select.options.length - 1;
                        if (newIndex >= select.options.length) newIndex = 0;
                        select.selectedIndex = newIndex;
                        select.dispatchEvent(new Event('change', { bubbles: true }));
                    }
                });
            }
        });
        const expirySelect = document.getElementById('expiry-select');
        const yfExpirySelect = document.getElementById('yf-expiry-select');
        expirySelect.addEventListener('change', () => {
            console.log('Expiry select changed to:', expirySelect.value);
            yfExpirySelect.value = expirySelect.value;
            updateCharts();
        });
        yfExpirySelect.addEventListener('change', () => {
            console.log('YF Expiry select changed to:', yfExpirySelect.value);
            expirySelect.value = yfExpirySelect.value;
            updateCharts();
        });
        document.getElementById('moneyness-select').addEventListener('change', () => {
            console.log('Moneyness select changed to:', document.getElementById('moneyness-select').value);
            updateCharts();
        });
        document.getElementById('history-select').addEventListener('change', () => {
            console.log('History length changed to:', document.getElementById('history-select').value);
            updateHistoricChart();
        });
        document.getElementById('realised-vol-history-select').addEventListener('change', () => {
            console.log('Realised Volatility history length changed to:', document.getElementById('realised-vol-history-select').value);
            updateRealisedVolChart();
        });
        document.getElementById('source-select').addEventListener('change', () => {
            console.log('Data source changed to:', document.getElementById('source-select').value);
            const dateSelect = document.getElementById('date-select');
            const timeSelect = document.getElementById('time-select');
            loadData(dateSelect.value + '_' + timeSelect.value);
        });
    } catch (error) {
        console.error('Error adding keyboard navigation:', error);
        document.getElementById('historic-error').textContent = `Error setting up keyboard navigation: ${error.message}`;
        document.getElementById('historic-error').style.display = 'block';
    }
}

function updateTimeOptions(selectedDate, dates) {
    try {
        const timeSelect = document.getElementById('time-select');
        timeSelect.innerHTML = '';
        const formattedSelectedDate = selectedDate.replace(/(\d{4})(\d{2})(\d{2})/, '$1-$2-$3');
        const availableTimes = dates
            .filter(date => date.replace(/(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})/, '$1-$2-$3') === formattedSelectedDate)
            .map(date => date.replace(/(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})/, '$4:$5'))
            .sort();
        const uniqueTimes = [...new Set(availableTimes)];
        uniqueTimes.forEach(timePart => {
            const option = document.createElement('option');
            option.value = timePart.replace(':', '');
            option.textContent = timePart;
            timeSelect.appendChild(option);
        });
        if (timeSelect.options.length > 0) {
            timeSelect.value = timeSelect.options[0].value;
            timeSelect.dispatchEvent(new Event('change', { bubbles: true }));
        } else {
            console.warn('No times available for selected date:', formattedSelectedDate);
            document.getElementById('historic-error').textContent = 'No times available for selected date';
            document.getElementById('historic-error').style.display = 'block';
        }
        console.log('Time options updated:', uniqueTimes);
    } catch (error) {
        console.error('Error updating time options:', error);
        document.getElementById('historic-error').textContent = `Error updating time options: ${error.message}`;
        document.getElementById('historic-error').style.display = 'block';
    }
}

document.addEventListener('DOMContentLoaded', () => {
    try {
        document.body.style.display = "none";
        checkPassword();
        if (!isAuthenticated) return;

        if (typeof Chart === 'undefined' || typeof Plotly === 'undefined' || typeof Papa === 'undefined' || typeof noUiSlider === 'undefined') {
            console.error('Required libraries not loaded');
            document.getElementById('historic-error').textContent = 'Failed to load required libraries';
            document.getElementById('historic-error').style.display = 'block';
            return;
        }

        const moneynessSlider = document.getElementById('moneyness-slider');
        const expiryTSlider = document.getElementById('expiry-t-slider');
        noUiSlider.create(moneynessSlider, {
            start: [0.6, 2.5],
            connect: true,
            range: { min: 0.1, max: 3 },
            step: 0.05
        });
        noUiSlider.create(expiryTSlider, {
            start: [0.2, 5],
            connect: true,
            range: { min: 0, max: 5 },
            step: 0.1
        });
        moneynessSlider.noUiSlider.on('update', (values) => {
            document.getElementById('moneyness-value').textContent = `${parseFloat(values[0]).toFixed(1)} - ${parseFloat(values[1]).toFixed(1)}`;
            updateCallVolSurface();
        });
        expiryTSlider.noUiSlider.on('update', (values) => {
            document.getElementById('expiry-t-value').textContent = `${parseFloat(values[0]).toFixed(1)} - ${parseFloat(values[1]).toFixed(1)}`;
            updateCallVolSurface();
        });

        const dateSelect = document.getElementById('date-select');
        const timeSelect = document.getElementById('time-select');
        const tickerSearch = document.getElementById('ticker-search');
        const sourceSelect = document.getElementById('source-select');

        const validateTimestamp = async (timestamp) => {
            const source = sourceSelect.value;
            const prefix = source === 'yfinance' ? '_yfinance' : '';
            try {
                const response = await fetch(`data/${timestamp}/tables/ranking/ranking_table${prefix}.csv?v=${Date.now()}`, { method: 'HEAD' });
                return response.ok ? timestamp : null;
            } catch {
                return null;
            }
        };

        const getLatestValidTimestamp = async (dates) => {
            const timestamps = dates.sort((a, b) => b.localeCompare(a));
            for (const ts of timestamps) {
                if (await validateTimestamp(ts)) return ts;
            }
            console.warn('No valid timestamps found, using fallback');
            return '20250827_2136';
        };

        fetch('data/dates.json?v=' + Date.now())
            .then(response => {
                if (!response.ok) throw new Error('Failed to load dates.json');
                return response.json();
            })
            .then(async dates => {
                const uniqueDates = [...new Set(dates.map(date => date.replace(/(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})/, '$1-$2-$3')))].sort((a, b) => b.localeCompare(a));
                dateSelect.innerHTML = '';
                uniqueDates.forEach(datePart => {
                    const option = document.createElement('option');
                    option.value = datePart.replace(/-/g, '');
                    option.textContent = datePart;
                    dateSelect.appendChild(option);
                });

                const loadInitialData = async () => {
                    let timestamp = await getLatestValidTimestamp(dates);
                    if (dateSelect.options.length > 0) {
                        dateSelect.value = timestamp.slice(0, 8);
                        await updateTimeOptions(dateSelect.value, dates);
                        timestamp = dateSelect.value + '_' + (timeSelect.options.length > 0 ? timeSelect.value : '2136');
                    }
                    console.log('Loading initial data with timestamp:', timestamp);
                    await loadData(timestamp);
                    updateDropdowns();
                    updateSection('overview');
                    console.log('Initial data and Overview loaded for timestamp:', timestamp);
                };

                await loadInitialData();

                dateSelect.addEventListener('change', async (e) => {
                    console.log('Date select changed to:', e.target.value);
                    await updateTimeOptions(e.target.value, dates);
                    const timestamp = e.target.value + '_' + (timeSelect.options.length > 0 ? timeSelect.value : '2136');
                    if (await validateTimestamp(timestamp)) {
                        await loadData(timestamp);
                        updateDropdowns();
                        const currentSection = document.querySelector('.overview-grid[style*="grid"], .volatility-grid[style*="grid"], .chart-container[style*="block"]');
                        const sectionId = currentSection ? currentSection.id : 'overview';
                        updateSection(sectionId);
                        console.log('Updated section after date change:', sectionId);
                    } else {
                        console.warn('Invalid timestamp:', timestamp);
                        document.getElementById('historic-error').textContent = 'Invalid date selected';
                        document.getElementById('historic-error').style.display = 'block';
                    }
                });

                timeSelect.addEventListener('change', async (e) => {
                    console.log('Time select changed to:', e.target.value);
                    const timestamp = dateSelect.value + '_' + e.target.value;
                    if (await validateTimestamp(timestamp)) {
                        await loadData(timestamp);
                        updateDropdowns();
                        const currentSection = document.querySelector('.overview-grid[style*="grid"], .volatility-grid[style*="grid"], .chart-container[style*="block"]');
                        const sectionId = currentSection ? currentSection.id : 'overview';
                        updateSection(sectionId);
                        console.log('Updated section after time change:', sectionId);
                    } else {
                        console.warn('Invalid timestamp:', timestamp);
                        document.getElementById('historic-error').textContent = 'Invalid time selected';
                        document.getElementById('historic-error').style.display = 'block';
                    }
                });

                tickerSearch.addEventListener('input', debounce(async () => {
                    console.log('Ticker changed to:', tickerSearch.value);
                    document.getElementById('ticker-display').textContent = `${tickerSearch.value.toUpperCase() || 'N/A'}`;
                    const timestamp = dateSelect.value + '_' + (timeSelect.options.length > 0 ? timeSelect.value : '2136');
                    if (await validateTimestamp(timestamp)) {
                        await loadData(timestamp);
                        updateDropdowns();
                        const currentSection = document.querySelector('.overview-grid[style*="grid"], .volatility-grid[style*="grid"], .chart-container[style*="block"]');
                        const sectionId = currentSection ? currentSection.id : 'overview';
                        updateSection(sectionId);
                        console.log('Updated section after ticker change:', sectionId);
                    }
                }, 500));
            })
            .catch(error => {
                console.error('Error loading dates:', error);
                document.getElementById('historic-error').textContent = `Error loading dates: ${error.message}`;
                document.getElementById('historic-error').style.display = 'block';
            });

        addKeyboardNavigation();
    } catch (error) {
        console.error('Error initializing page:', error);
        document.getElementById('historic-error').textContent = `Error initializing page: ${error.message}`;
        document.getElementById('historic-error').style.display = 'block';
    }
});
