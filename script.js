let historicData = [];
let rankingTableData = [];
let stockTableData = [];
let summaryTableData = [];
let topVolumeData = [];
let topOpenInterestData = [];
let volSurfaceData = [];
let skewData = [];
let cleanedData = [];
let eventsData = [];
let companyNames = [];
let moneynessChart, expiryChart, yfIVChart, skewChart, realisedVolChart, historicChart, volumeChart, openInterestChart;
let statsTablesVisible = false;
let rankingTableVisible = false;
let stockTableVisible = false;
let lastLoadDataTimestamp = null;
let loadDataTimeout = null;
let lastTicker = null;

function parseCSV(csvText) {
    try {
        const result = Papa.parse(csvText, {
            header: true,
            skipEmptyLines: true,
            dynamicTyping: true
        });
        return result.data;
    } catch (error) {
        console.error('Error parsing CSV:', error);
        return [];
    }
}

async function fetchWithErrorHandling(url, array) {
    try {
        const response = await fetch(`${url}?v=${Date.now()}`);
        if (!response.ok) throw new Error(`Failed to fetch ${url}: ${response.statusText}`);
        const parsedData = parseCSV(await response.text());
        array.splice(0, array.length, ...parsedData);
        console.log(`Data loaded from ${url}, size:`, array.length, 'sample:', array.slice(0, 5));
    } catch (error) {
        console.warn(`Error loading ${url}:`, error);
        array.splice(0, array.length);
        document.getElementById('historic-error').textContent = `Failed to load ${url}: ${error.message}`;
        document.getElementById('historic-error').style.display = 'block';
    }
}

function normalizeDate(dateStr) {
    try {
        const date = new Date(dateStr);
        if (isNaN(date.getTime())) throw new Error('Invalid date');
        return date.toLocaleDateString('en-CA');
    } catch (error) {
        console.error('Error normalizing date:', dateStr, error);
        return null;
    }
}

function calculateDateRange(historyLength) {
    try {
        const now = new Date();
        let fromDate = new Date();
        switch (historyLength) {
            case '1m': fromDate.setMonth(now.getMonth() - 1); break;
            case '3m': fromDate.setMonth(now.getMonth() - 3); break;
            case '6m': fromDate.setMonth(now.getMonth() - 6); break;
            case '1y': fromDate.setFullYear(now.getFullYear() - 1); break;
            case '5y': fromDate.setFullYear(now.getFullYear() - 5); break;
            case 'all': default: return null;
        }
        return fromDate;
    } catch (error) {
        console.error('Error calculating date range:', error);
        return null;
    }
}

function changeSelect(selectId, direction) {
    try {
        const select = document.getElementById(selectId);
        let newIndex = select.selectedIndex + direction;
        if (newIndex < 0) newIndex = select.options.length - 1;
        if (newIndex >= select.options.length) newIndex = 0;
        select.selectedIndex = newIndex;
        select.dispatchEvent(new Event('change', { bubbles: true }));
        console.log(`Changed ${selectId} to:`, select.value);
    } catch (error) {
        console.error('Error in changeSelect:', error);
    }
}

function toggleStatsTables() {
    try {
        statsTablesVisible = !statsTablesVisible;
        document.querySelectorAll('.stats-table:not(#historic-price-chart + .stats-table)').forEach(table => {
            table.style.display = statsTablesVisible ? 'table' : 'none';
        });
        document.querySelector('.toggle-button:not(#ranking .toggle-button)').textContent = statsTablesVisible ? 'Hide Stats Tables' : 'Show Stats Tables';
        console.log(`Stats tables visibility set to: ${statsTablesVisible}`);
    } catch (error) {
        console.error('Error in toggleStatsTables:', error);
    }
}

function toggleRankingTable() {
    try {
        rankingTableVisible = !rankingTableVisible;
        const table = document.getElementById('ranking-table');
        table.style.display = rankingTableVisible ? 'table' : 'none';
        document.querySelector('#ranking .toggle-button').textContent = rankingTableVisible ? 'Hide Data' : 'Show Data';
        if (rankingTableVisible) updateRankingTable();
        console.log(`Ranking table visibility set to: ${rankingTableVisible}`);
    } catch (error) {
        console.error('Error in toggleRankingTable:', error);
    }
}

function toggleStockTable() {
    try {
        stockTableVisible = !stockTableVisible;
        const table = document.getElementById('stock-table');
        table.style.display = stockTableVisible ? 'table' : 'none';
        document.querySelector('#stock .toggle-button').textContent = stockTableVisible ? 'Hide Data' : 'Show Data';
        if (stockTableVisible) updateStockTable();
        console.log(`Stock table visibility set to: ${stockTableVisible}`);
    } catch (error) {
        console.error('Error in toggleStockTable:', error);
    }
}

function debounce(func, wait) {
    let timeout;
    return function (...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), wait);
    };
}

async function loadData(timestamp) {
    try {
        const selectedTicker = document.getElementById('ticker-search').value.toUpperCase() || 'MSTR';
        if (lastLoadDataTimestamp === timestamp && lastTicker === selectedTicker) {
            console.log(`Skipping redundant loadData for timestamp: ${timestamp}, ticker: ${selectedTicker}`);
            return Promise.resolve();
        }
        lastLoadDataTimestamp = timestamp;
        lastTicker = selectedTicker;
        clearTimeout(loadDataTimeout);
        document.getElementById('historic-error').textContent = 'Loading data...';
        document.getElementById('historic-error').style.display = 'block';
        const source = document.getElementById('source-select').value;
        const prefix = source === 'yfinance' ? '_yfinance' : '';
        console.log(`Loading data for timestamp: ${timestamp}, source: ${source}, ticker: ${selectedTicker}`);

        // Clear existing data
        historicData = [];
        rankingTableData = [];
        stockTableData = [];
        summaryTableData = [];
        topVolumeData = [];
        topOpenInterestData = [];
        volSurfaceData = [];
        skewData = [];
        cleanedData = [];
        eventsData = [];
        companyNames = [];

        const checkFileExists = async (url) => {
            try {
                const response = await fetch(url, { method: 'HEAD' });
                return response.ok;
            } catch {
                return false;
            }
        };

        // Load precomputed data
        const [
            companyNamesExists,
            rankingTableExists,
            stockTableExists,
            summaryExists,
            topVolumeExists,
            topOpenInterestExists,
            volSurfaceExists,
            skewExists,
            cleanedExists,
            historicExists,
            eventsExists
        ] = await Promise.all([
            checkFileExists(`./company_names.txt`),
            checkFileExists(`data/${timestamp}/tables/ranking/ranking_table${prefix}.csv`),
            checkFileExists(`data/${timestamp}/tables/stock/stock_table${prefix}.csv`),
            checkFileExists(`data/${timestamp}/tables/summary/summary${prefix}_${selectedTicker}.csv`),
            checkFileExists(`data/${timestamp}/tables/contracts/${selectedTicker}/top_volume${prefix}_${selectedTicker}.csv`),
            checkFileExists(`data/${timestamp}/tables/contracts/${selectedTicker}/top_open_interest${prefix}_${selectedTicker}.csv`),
            checkFileExists(`data/${timestamp}/tables/vol_surface/vol_surface${prefix}_${selectedTicker}.csv`),
            checkFileExists(`data/${timestamp}/skew_metrics${prefix}/skew_metrics${prefix}_${selectedTicker}.csv`),
            checkFileExists(`data/${timestamp}/cleaned_yfinance/cleaned_yfinance_${selectedTicker}.csv`),
            checkFileExists(`data/${timestamp}/historic/historic_${selectedTicker}.csv`),
            checkFileExists(`data/Events.csv`)
        ]);

        await Promise.all([
            companyNamesExists ? fetchWithErrorHandling(`./company_names.txt`, companyNames) : Promise.resolve(),
            rankingTableExists ? fetchWithErrorHandling(`data/${timestamp}/tables/ranking/ranking_table${prefix}.csv`, rankingTableData) : Promise.resolve(),
            stockTableExists ? fetchWithErrorHandling(`data/${timestamp}/tables/stock/stock_table${prefix}.csv`, stockTableData) : Promise.resolve(),
            summaryExists ? fetchWithErrorHandling(`data/${timestamp}/tables/summary/summary${prefix}_${selectedTicker}.csv`, summaryTableData) : Promise.resolve(),
            topVolumeExists ? fetchWithErrorHandling(`data/${timestamp}/tables/contracts/${selectedTicker}/top_volume${prefix}_${selectedTicker}.csv`, topVolumeData) : Promise.resolve(),
            topOpenInterestExists ? fetchWithErrorHandling(`data/${timestamp}/tables/contracts/${selectedTicker}/top_open_interest${prefix}_${selectedTicker}.csv`, topOpenInterestData) : Promise.resolve(),
            volSurfaceExists ? fetchWithErrorHandling(`data/${timestamp}/tables/vol_surface/vol_surface${prefix}_${selectedTicker}.csv`, volSurfaceData) : Promise.resolve(),
            skewExists ? fetchWithErrorHandling(`data/${timestamp}/skew_metrics${prefix}/skew_metrics${prefix}_${selectedTicker}.csv`, skewData) : Promise.resolve(),
            cleanedExists ? fetchWithErrorHandling(`data/${timestamp}/cleaned_yfinance/cleaned_yfinance_${selectedTicker}.csv`, cleanedData) : Promise.resolve(),
            historicExists ? fetchWithErrorHandling(`data/${timestamp}/historic/historic_${selectedTicker}.csv`, historicData).then(() => {
                document.getElementById('historic-error').style.display = historicData.length ? 'none' : 'block';
                if (!historicData.length) document.getElementById('historic-error').textContent = 'No historic data available';
            }) : Promise.resolve(),
            eventsExists ? fetchWithErrorHandling(`data/Events.csv`, eventsData) : Promise.resolve()
        ]);

        // Populate ticker datalist
        const tickerDatalist = document.getElementById('ticker-datalist');
        tickerDatalist.innerHTML = '';
        const uniqueTickers = [...new Set(rankingTableData.map(item => item.Ticker))].filter(val => val).sort();
        uniqueTickers.forEach(val => {
            const option = document.createElement('option');
            option.value = val;
            tickerDatalist.appendChild(option);
        });

        // Validate selected ticker
        if (!uniqueTickers.includes(selectedTicker)) {
            console.warn(`Selected ticker ${selectedTicker} not in ranking data, defaulting`);
            const defaultTicker = uniqueTickers.includes('MSTR') ? 'MSTR' : uniqueTickers[0] || '';
            document.getElementById('ticker-search').value = defaultTicker;
            document.getElementById('ticker-display').textContent = `${defaultTicker || 'N/A'}`;
            if (!defaultTicker) {
                console.warn('No valid tickers available');
                document.getElementById('historic-error').textContent = 'No valid tickers available';
                return Promise.resolve();
            }
        } else {
            document.getElementById('ticker-display').textContent = `${selectedTicker || 'N/A'}`;
        }

        document.getElementById('historic-error').style.display = 'none';
        return Promise.resolve();
    } catch (error) {
        console.error('Error in loadData:', error);
        document.getElementById('ticker-search').value = '';
        document.getElementById('ticker-datalist').innerHTML = '';
        document.getElementById('ticker-display').textContent = 'N/A';
        document.getElementById('historic-error').textContent = `Error loading data: ${error.message}`;
        document.getElementById('historic-error').style.display = 'block';
        return Promise.resolve();
    }
}

function updateDropdowns() {
    try {
        const tickerSearch = document.getElementById('ticker-search');
        const selectedTicker = tickerSearch.value.toUpperCase() || 'MSTR';
        const moneynessSelect = document.getElementById('moneyness-select');
        const expirySelect = document.getElementById('expiry-select');
        const yfExpirySelect = document.getElementById('yf-expiry-select');

        // Populate moneyness dropdown from volSurfaceData
        const uniqueRoundedMoneyness = [...new Set(volSurfaceData.map(item => item.Moneyness))].filter(val => !isNaN(val)).sort((a, b) => a - b);
        moneynessSelect.innerHTML = '';
        if (uniqueRoundedMoneyness.length === 0) {
            moneynessSelect.innerHTML = '<option value="">No moneyness available</option>';
        } else {
            uniqueRoundedMoneyness.forEach(val => {
                const option = document.createElement('option');
                option.value = val;
                option.text = `${(val * 100).toFixed(0)}%`;
                moneynessSelect.appendChild(option);
            });
            const targetMoneyness = 1.0;
            let closestMoneyness = uniqueRoundedMoneyness[0];
            if (uniqueRoundedMoneyness.length > 0) {
                closestMoneyness = uniqueRoundedMoneyness.reduce((prev, curr) =>
                    Math.abs(curr - targetMoneyness) < Math.abs(prev - targetMoneyness) ? curr : prev
                );
            }
            moneynessSelect.value = closestMoneyness.toString();
        }

        // Populate expiry dropdowns from volSurfaceData
        const uniqueExpiries = [...new Set(volSurfaceData.map(item => item.Expiry))]
            .filter(val => val && !isNaN(new Date(val).getTime())).sort((a, b) => new Date(a) - new Date(b));
        expirySelect.innerHTML = '';
        yfExpirySelect.innerHTML = '';
        if (uniqueExpiries.length === 0) {
            expirySelect.innerHTML = '<option value="">No expiries available</option>';
            yfExpirySelect.innerHTML = '<option value="">No expiries available</option>';
        } else {
            uniqueExpiries.forEach(val => {
                const option = document.createElement('option');
                option.value = val;
                option.text = new Date(val).toLocaleDateString('en-GB');
                expirySelect.appendChild(option);
                yfExpirySelect.appendChild(option.cloneNode(true));
            });
            expirySelect.value = uniqueExpiries[0];
            yfExpirySelect.value = uniqueExpiries[0];
        }
    } catch (error) {
        console.error('Error updating dropdowns:', error);
        document.getElementById('historic-error').textContent = `Error updating dropdowns: ${error.message}`;
        document.getElementById('historic-error').style.display = 'block';
    }
}

function updateSummaryTable() {
    try {
        const table = document.getElementById('summary-table');
        table.innerHTML = '';
        if (!summaryTableData.length) {
            table.innerHTML = '<tr><td colspan="2">No summary data available</td></tr>';
            console.warn('No summary data for table');
            return;
        }
        const thead = document.createElement('thead');
        const headerRow = thead.insertRow();
        ['Metric', 'Value'].forEach(text => {
            const th = document.createElement('th');
            th.textContent = text;
            th.style.textAlign = text === 'Metric' ? 'left' : 'right';
            th.style.color = '#FFFFFF';
            headerRow.appendChild(th);
        });
        table.appendChild(thead);
        const tbody = document.createElement('tbody');
        summaryTableData.forEach(item => {
            const row = tbody.insertRow();
            const metricCell = row.insertCell();
            metricCell.textContent = item.Metric;
            metricCell.style.textAlign = 'left';
            metricCell.style.color = '#FFFFFF';
            const valueCell = row.insertCell();
            valueCell.textContent = item.Value;
            valueCell.style.textAlign = 'right';
            valueCell.style.color = item.Color || '#FFFFFF';
        });
        table.appendChild(tbody);
        console.log('Summary table updated');
    } catch (error) {
        console.error('Error updating summary table:', error);
        document.getElementById('summary-table').innerHTML = '<tr><td colspan="2">Error loading summary table</td></tr>';
    }
}

function updateRankingTable() {
    try {
        const table = document.getElementById('ranking-table');
        table.innerHTML = '';
        if (!rankingTableData.length) {
            table.innerHTML = '<tr><td colspan="30">No ranking data available</td></tr>';
            console.warn('No ranking data for table');
            return;
        }
        const thead = document.createElement('thead');
        const headerRow = thead.insertRow();
        const columns = Object.keys(rankingTableData[0]).filter(col => !col.endsWith('_Color'));
        columns.forEach((text, index) => {
            const th = document.createElement('th');
            th.textContent = text;
            th.dataset.column = text;
            th.dataset.order = 'desc';
            th.style.color = '#FFFFFF';
            th.addEventListener('click', () => sortTable('ranking-table', index));
            headerRow.appendChild(th);
        });
        table.appendChild(thead);
        const tbody = document.createElement('tbody');
        rankingTableData.forEach(item => {
            const row = tbody.insertRow();
            columns.forEach(col => {
                const cell = row.insertCell();
                cell.textContent = item[col] || 'N/A';
                cell.style.color = item[`${col}_Color`] || '#FFFFFF';
            });
        });
        table.appendChild(tbody);
        console.log('Ranking table updated');
    } catch (error) {
        console.error('Error updating ranking table:', error);
        document.getElementById('ranking-table').innerHTML = '<tr><td colspan="30">Error loading ranking table</td></tr>';
    }
}

function updateStockTable() {
    try {
        const table = document.getElementById('stock-table');
        table.innerHTML = '';
        if (!stockTableData.length) {
            table.innerHTML = '<tr><td colspan="18">No stock data available</td></tr>';
            console.warn('No stock data for table');
            return;
        }
        const thead = document.createElement('thead');
        const headerRow = thead.insertRow();
        const columns = Object.keys(stockTableData[0]).filter(col => !col.endsWith('_Color'));
        columns.forEach((text, index) => {
            const th = document.createElement('th');
            th.textContent = text;
            th.dataset.column = text;
            th.dataset.order = 'desc';
            th.style.color = '#FFFFFF';
            th.addEventListener('click', () => sortTable('stock-table', index));
            headerRow.appendChild(th);
        });
        table.appendChild(thead);
        const tbody = document.createElement('tbody');
        stockTableData.forEach(item => {
            const row = tbody.insertRow();
            columns.forEach(col => {
                const cell = row.insertCell();
                cell.textContent = item[col] || 'N/A';
                cell.style.color = item[`${col}_Color`] || '#FFFFFF';
            });
        });
        table.appendChild(tbody);
        console.log('Stock table updated');
    } catch (error) {
        console.error('Error updating stock table:', error);
        document.getElementById('stock-table').innerHTML = '<tr><td colspan="18">Error loading stock table</td></tr>';
    }
}

function updateTopContractsTables() {
    try {
        const volumeTable = document.getElementById('top-volume-table');
        const openInterestTable = document.getElementById('top-open-interest-table');
        volumeTable.innerHTML = '';
        openInterestTable.innerHTML = '';

        const createTable = (table, data, title) => {
            const thead = document.createElement('thead');
            const headerRow = thead.insertRow();
            ['Strike', 'Expiry', 'Type', 'Bid', 'Ask', 'Volume', 'Open Interest'].forEach(text => {
                const th = document.createElement('th');
                th.textContent = text;
                headerRow.appendChild(th);
            });
            table.appendChild(thead);
            const tbody = document.createElement('tbody');
            if (!data.length) {
                const row = tbody.insertRow();
                const cell = row.insertCell();
                cell.textContent = `No ${title.toLowerCase()} data available`;
                cell.colSpan = 7;
            } else {
                data.forEach(item => {
                    const row = tbody.insertRow();
                    ['Strike', 'Expiry', 'Type', 'Bid', 'Ask', 'Volume', 'Open Interest'].forEach(col => {
                        const cell = row.insertCell();
                        cell.textContent = item[col] || 'N/A';
                    });
                });
            }
            table.appendChild(tbody);
        };

        createTable(volumeTable, topVolumeData, 'Volume');
        createTable(openInterestTable, topOpenInterestData, 'Open Interest');
        console.log('Top contracts tables updated');
    } catch (error) {
        console.error('Error updating top contracts tables:', error);
        document.getElementById('top-volume-table').innerHTML = '<tr><td colspan="7">Error loading volume table</td></tr>';
        document.getElementById('top-open-interest-table').innerHTML = '<tr><td colspan="7">Error loading open interest table</td></tr>';
    }
}

function updateMoneynessVsIV() {
    try {
        const selectedTicker = document.getElementById('ticker-search').value.toUpperCase();
        const selectedExpiry = document.getElementById('expiry-select').value;
        const filteredData = cleanedData.filter(item => item.Ticker === selectedTicker && item.Expiry === selectedExpiry);
        const errorDiv = document.getElementById('moneyness-error');
        errorDiv.style.display = 'none';
        if (!filteredData.length) {
            if (moneynessChart) moneynessChart.destroy();
            errorDiv.textContent = 'No data for selected ticker and expiry';
            errorDiv.style.display = 'block';
            return;
        }
        const datasets = [
            {
                label: 'Calls Smoothed IV',
                data: filteredData.filter(item => item.Type === 'Call').map(item => ({ x: item.Moneyness * 100, y: item.Smoothed_IV * 100 })),
                borderColor: '#FFFF00',
                backgroundColor: 'rgba(255, 255, 0, 0.2)',
                pointRadius: 3,
                pointStyle: 'circle',
                fill: true,
                showLine: true,
                yAxisID: 'y'
            },
            {
                label: 'Puts Smoothed IV',
                data: filteredData.filter(item => item.Type === 'Put').map(item => ({ x: item.Moneyness * 100, y: item.Smoothed_IV * 100 })),
                borderColor: '#26A69A',
                backgroundColor: 'rgba(38, 166, 154, 0.2)',
                pointRadius: 3,
                pointStyle: 'circle',
                fill: true,
                showLine: true,
                yAxisID: 'y'
            }
        ];
        const moneynessValues = filteredData
            .filter(item => Number.isFinite(item.Moneyness))
            .map(item => item.Moneyness * 100);
        const minMoneyness = moneynessValues.length > 0 ? Math.min(...moneynessValues) : 0;
        const maxMoneyness = moneynessValues.length > 0 ? Math.max(...moneynessValues) : 200;
        if (moneynessChart) moneynessChart.destroy();
        const canvas = document.getElementById('moneyness-vs-iv-chart');
        if (!canvas.getContext('2d')) throw new Error('Canvas context not available');
        moneynessChart = new Chart(canvas.getContext('2d'), {
            type: 'line',
            data: { datasets },
            options: {
                plugins: { legend: { labels: { color: '#FFFFFF' } } },
                scales: {
                    x: {
                        type: 'linear',
                        title: { display: true, text: 'Moneyness (%)', color: '#FFFFFF' },
                        min: minMoneyness,
                        max: maxMoneyness,
                        grid: { display: false },
                        ticks: { color: '#FFFFFF' }
                    },
                    y: {
                        title: { display: true, text: 'Volatility (%)', color: '#FFFFFF' },
                        beginAtZero: true,
                        grid: { display: false },
                        ticks: { color: '#FFFFFF' }
                    }
                },
                backgroundColor: '#1A1A2E'
            }
        });
        createStatsTable(moneynessChart, 'moneyness-vs-iv-chart');
    } catch (error) {
        console.error('Error updating Moneyness vs. IV chart:', error);
        document.getElementById('moneyness-error').textContent = `Error rendering chart: ${error.message}`;
        document.getElementById('moneyness-error').style.display = 'block';
    }
}

function updateExpiryVsIV() {
    try {
        const selectedTicker = document.getElementById('ticker-search').value.toUpperCase();
        const selectedMoneyness = parseFloat(document.getElementById('moneyness-select').value);
        const filteredData = cleanedData.filter(item => item.Ticker === selectedTicker && Math.round(item.Moneyness * 100 / 10) * 0.1 === selectedMoneyness);
        const errorDiv = document.getElementById('expiry-error');
        errorDiv.style.display = 'none';
        if (!filteredData.length) {
            if (expiryChart) expiryChart.destroy();
            errorDiv.textContent = 'No data for selected ticker and moneyness';
            errorDiv.style.display = 'block';
            return;
        }
        const datasets = [
            {
                label: 'Calls Smoothed IV',
                data: filteredData.filter(item => item.Type === 'Call').map(item => ({ x: new Date(item.Expiry), y: item.Smoothed_IV * 100 })),
                borderColor: '#FFFF00',
                backgroundColor: 'rgba(255, 255, 0, 0.2)',
                pointRadius: 3,
                pointStyle: 'circle',
                fill: true,
                showLine: true,
                yAxisID: 'y'
            },
            {
                label: 'Puts Smoothed IV',
                data: filteredData.filter(item => item.Type === 'Put').map(item => ({ x: new Date(item.Expiry), y: item.Smoothed_IV * 100 })),
                borderColor: '#26A69A',
                backgroundColor: 'rgba(38, 166, 154, 0.2)',
                pointRadius: 3,
                pointStyle: 'circle',
                fill: true,
                showLine: true,
                yAxisID: 'y'
            }
        ];
        if (expiryChart) expiryChart.destroy();
        const canvas = document.getElementById('expiry-vs-iv-chart');
        if (!canvas.getContext('2d')) throw new Error('Canvas context not available');
        expiryChart = new Chart(canvas.getContext('2d'), {
            type: 'line',
            data: { datasets },
            options: {
                plugins: { legend: { labels: { color: '#FFFFFF' } } },
                scales: {
                    x: {
                        type: 'time',
                        time: { unit: 'day' },
                        title: { display: true, text: 'Expiry Date', color: '#FFFFFF' },
                        grid: { display: false },
                        ticks: { color: '#FFFFFF' }
                    },
                    y: {
                        title: { display: true, text: 'Volatility (%)', color: '#FFFFFF' },
                        beginAtZero: true,
                        grid: { display: false },
                        ticks: { color: '#FFFFFF' }
                    }
                },
                backgroundColor: '#1A1A2E'
            }
        });
        createStatsTable(expiryChart, 'expiry-vs-iv-chart');
    } catch (error) {
        console.error('Error updating Expiry vs. IV chart:', error);
        document.getElementById('expiry-error').textContent = `Error rendering chart: ${error.message}`;
        document.getElementById('expiry-error').style.display = 'block';
    }
}

function updateYFIVChart() {
    try {
        const selectedTicker = document.getElementById('ticker-search').value.toUpperCase();
        const selectedExpiry = document.getElementById('yf-expiry-select').value;
        const filteredData = cleanedData.filter(item => item.Ticker === selectedTicker && item.Expiry === selectedExpiry);
        const errorDiv = document.getElementById('yf-iv-error');
        errorDiv.style.display = 'none';
        if (!filteredData.length) {
            if (yfIVChart) yfIVChart.destroy();
            errorDiv.textContent = 'No data for selected ticker and expiry';
            errorDiv.style.display = 'block';
            return;
        }
        const datasets = [
            {
                label: 'Calls Implied Volatility',
                data: filteredData.filter(item => item.Type === 'Call').map(item => ({ x: item.Moneyness * 100, y: item['Implied Volatility'] * 100 })),
                borderColor: '#FFFF00',
                backgroundColor: 'rgba(255, 255, 0, 0.2)',
                pointRadius: 3,
                pointStyle: 'circle',
                fill: true,
                showLine: true,
                yAxisID: 'y'
            },
            {
                label: 'Puts Implied Volatility',
                data: filteredData.filter(item => item.Type === 'Put').map(item => ({ x: item.Moneyness * 100, y: item['Implied Volatility'] * 100 })),
                borderColor: '#26A69A',
                backgroundColor: 'rgba(38, 166, 154, 0.2)',
                pointRadius: 3,
                pointStyle: 'circle',
                fill: true,
                showLine: true,
                yAxisID: 'y'
            }
        ];
        const moneynessValues = filteredData
            .filter(item => Number.isFinite(item.Moneyness))
            .map(item => item.Moneyness * 100);
        const minMoneyness = moneynessValues.length > 0 ? Math.min(...moneynessValues) : 0;
        const maxMoneyness = moneynessValues.length > 0 ? Math.max(...moneynessValues) : 200;
        if (yfIVChart) yfIVChart.destroy();
        const canvas = document.getElementById('yf-iv-chart');
        if (!canvas.getContext('2d')) throw new Error('Canvas context not available');
        yfIVChart = new Chart(canvas.getContext('2d'), {
            type: 'line',
            data: { datasets },
            options: {
                plugins: { legend: { labels: { color: '#FFFFFF' } } },
                scales: {
                    x: {
                        type: 'linear',
                        title: { display: true, text: 'Moneyness (%)', color: '#FFFFFF' },
                        min: minMoneyness,
                        max: maxMoneyness,
                        grid: { display: false },
                        ticks: { color: '#FFFFFF' }
                    },
                    y: {
                        title: { display: true, text: 'Implied Volatility (%)', color: '#FFFFFF' },
                        beginAtZero: true,
                        grid: { display: false },
                        ticks: { color: '#FFFFFF' }
                    }
                },
                backgroundColor: '#1A1A2E'
            }
        });
        createStatsTable(yfIVChart, 'yf-iv-chart');
    } catch (error) {
        console.error('Error updating Yahoo Finance IV chart:', error);
        document.getElementById('yf-iv-error').textContent = `Error rendering chart: ${error.message}`;
        document.getElementById('yf-iv-error').style.display = 'block';
    }
}

function updateSkewVsExpiry() {
    try {
        const selectedTicker = document.getElementById('ticker-search').value.toUpperCase();
        const filteredSkewData = skewData.filter(item => item.Ticker && item.Ticker.toUpperCase() === selectedTicker);
        const errorDiv = document.getElementById('skew-error');
        errorDiv.style.display = 'none';
        if (!filteredSkewData.length) {
            if (skewChart) skewChart.destroy();
            errorDiv.textContent = 'No skew data available';
            errorDiv.style.display = 'block';
            return;
        }
        const datasets = [
            {
                label: 'Skew 25 Delta',
                data: filteredSkewData.map(item => ({ x: new Date(item.Expiry), y: item.Skew_25_delta })),
                borderColor: '#FFA500',
                backgroundColor: 'rgba(255, 165, 0, 0.2)',
                pointRadius: 3,
                pointStyle: 'circle',
                fill: true,
                showLine: true,
                yAxisID: 'y'
            },
            {
                label: 'Skew 75 Delta',
                data: filteredSkewData.map(item => ({ x: new Date(item.Expiry), y: item.Skew_75_delta })),
                borderColor: '#00FF00',
                backgroundColor: 'rgba(0, 255, 0, 0.2)',
                pointRadius: 3,
                pointStyle: 'circle',
                fill: true,
                showLine: true,
                yAxisID: 'y'
            },
            {
                label: 'Skew Call 25/75',
                data: filteredSkewData.map(item => ({ x: new Date(item.Expiry), y: item.Skew_call_25_75 })),
                borderColor: '#FF0000',
                backgroundColor: 'rgba(255, 0, 0, 0.2)',
                pointRadius: 3,
                pointStyle: 'circle',
                fill: true,
                showLine: true,
                yAxisID: 'y'
            },
            {
                label: 'Skew Put 25/75',
                data: filteredSkewData.map(item => ({ x: new Date(item.Expiry), y: item.Skew_put_25_75 })),
                borderColor: '#26A69A',
                backgroundColor: 'rgba(38, 166, 154, 0.2)',
                pointRadius: 3,
                pointStyle: 'circle',
                fill: true,
                showLine: true,
                yAxisID: 'y'
            }
        ];
        if (skewChart) skewChart.destroy();
        const canvas = document.getElementById('skew-vs-expiry-chart');
        if (!canvas.getContext('2d')) throw new Error('Canvas context not available');
        skewChart = new Chart(canvas.getContext('2d'), {
            type: 'line',
            data: { datasets },
            options: {
                plugins: { legend: { labels: { color: '#FFFFFF' } } },
                scales: {
                    x: {
                        type: 'time',
                        time: { unit: 'day' },
                        title: { display: true, text: 'Expiry Date', color: '#FFFFFF' },
                        grid: { display: false },
                        ticks: { color: '#FFFFFF' }
                    },
                    y: {
                        title: { display: true, text: 'Skew', color: '#FFFFFF' },
                        beginAtZero: false,
                        grid: { display: false },
                        ticks: { color: '#FFFFFF' }
                    }
                },
                backgroundColor: '#1A1A2E'
            }
        });
        createStatsTable(skewChart, 'skew-vs-expiry-chart');
    } catch (error) {
        console.error('Error updating Skew vs Expiry chart:', error);
        document.getElementById('skew-error').textContent = `Error rendering chart: ${error.message}`;
        document.getElementById('skew-error').style.display = 'block';
    }
}

function updateRealisedVolChart() {
    try {
        const selectedTicker = document.getElementById('ticker-search').value.toUpperCase();
        const historyLength = document.getElementById('realised-vol-history-select').value;
        const errorDiv = document.getElementById('realised-vol-error');
        errorDiv.style.display = 'none';
        let filteredHistoric = historicData.filter(item => item.Ticker === selectedTicker && item.Date);
        if (!filteredHistoric.length) {
            if (realisedVolChart) realisedVolChart.destroy();
            errorDiv.textContent = 'No historic data available';
            errorDiv.style.display = 'block';
            return;
        }
        filteredHistoric.sort((a, b) => new Date(a.Date) - new Date(b.Date));
        const fromDate = calculateDateRange(historyLength);
        if (fromDate) {
            filteredHistoric = filteredHistoric.filter(item => new Date(item.Date) >= fromDate);
        }
        const datasets = [
            {
                label: '30-day Realised Vol (%)',
                data: filteredHistoric.map(item => ({ x: new Date(item.Date), y: item['Realised_Vol_Close_30'] || item['Realized_Vol_Close_30'] })),
                borderColor: '#FFFF00',
                backgroundColor: 'rgba(255, 255, 0, 0.2)',
                pointRadius: 1,
                fill: true,
                showLine: true,
                yAxisID: 'y'
            },
            {
                label: '60-day Realised Vol (%)',
                data: filteredHistoric.map(item => ({ x: new Date(item.Date), y: item['Realised_Vol_Close_60'] || item['Realized_Vol_Close_60'] })),
                borderColor: '#008000',
                backgroundColor: 'rgba(0, 128, 0, 0.2)',
                pointRadius: 1,
                fill: true,
                showLine: true,
                yAxisID: 'y'
            },
            {
                label: '100-day Realised Vol (%)',
                data: filteredHistoric.map(item => ({ x: new Date(item.Date), y: item['Realised_Vol_Close_100'] || item['Realized_Vol_Close_100'] })),
                borderColor: '#FF0000',
                backgroundColor: 'rgba(255, 0, 0, 0.2)',
                pointRadius: 1,
                fill: true,
                showLine: true,
                yAxisID: 'y'
            },
            {
                label: '180-day Realised Vol (%)',
                data: filteredHistoric.map(item => ({ x: new Date(item.Date), y: item['Realised_Vol_Close_180'] || item['Realized_Vol_Close_180'] })),
                borderColor: '#FFA500',
                backgroundColor: 'rgba(255, 165, 0, 0.2)',
                pointRadius: 1,
                fill: true,
                showLine: true,
                yAxisID: 'y'
            },
            {
                label: '252-day Realised Vol (%)',
                data: filteredHistoric.map(item => ({ x: new Date(item.Date), y: item['Realised_Vol_Close_252'] || item['Realized_Vol_Close_252'] })),
                borderColor: '#26A69A',
                backgroundColor: 'rgba(38, 166, 154, 0.2)',
                pointRadius: 1,
                fill: true,
                showLine: true,
                yAxisID: 'y'
            }
        ].filter(ds => ds.data.some(point => point.y !== null && Number.isFinite(point.y)));
        if (realisedVolChart) realisedVolChart.destroy();
        const canvas = document.getElementById('realised-vol-chart');
        if (!canvas.getContext('2d')) throw new Error('Canvas context not available');
        realisedVolChart = new Chart(canvas.getContext('2d'), {
            type: 'line',
            data: { datasets },
            options: {
                plugins: { legend: { labels: { color: '#FFFFFF' } } },
                scales: {
                    x: {
                        type: 'time',
                        time: { unit: 'month' },
                        title: { display: true, text: 'Date', color: '#FFFFFF' },
                        grid: { display: false },
                        ticks: { color: '#FFFFFF' }
                    },
                    y: {
                        title: { display: true, text: 'Realised Volatility (%)', color: '#FFFFFF' },
                        beginAtZero: false,
                        grid: { display: false },
                        ticks: { color: '#FFFFFF' }
                    }
                },
                backgroundColor: '#1A1A2E'
            }
        });
        createStatsTable(realisedVolChart, 'realised-vol-chart');
    } catch (error) {
        console.error('Error updating Realised Volatility chart:', error);
        document.getElementById('realised-vol-error').textContent = `Error rendering chart: ${error.message}`;
        document.getElementById('realised-vol-error').style.display = 'block';
    }
}

function updateCallVolSurface() {
    try {
        const selectedTicker = document.getElementById('ticker-search').value.toUpperCase();
        const moneynessSlider = document.getElementById('moneyness-slider');
        const expiryTSlider = document.getElementById('expiry-t-slider');
        const moneynessMin = parseFloat(moneynessSlider.noUiSlider.get()[0]);
        const moneynessMax = parseFloat(moneynessSlider.noUiSlider.get()[1]);
        const expiryTMin = parseFloat(expiryTSlider.noUiSlider.get()[0]);
        const expiryTMax = parseFloat(expiryTSlider.noUiSlider.get()[1]);
        const errorDiv = document.getElementById('surface-error');
        errorDiv.style.display = 'none';
        const filteredData = volSurfaceData.filter(item => 
            item.Moneyness >= moneynessMin && 
            item.Moneyness <= moneynessMax && 
            item.Expiry_T >= expiryTMin && 
            item.Expiry_T <= expiryTMax
        );
        if (!filteredData.length) {
            Plotly.purge('call-vol-surface');
            errorDiv.textContent = 'No data for selected moneyness and expiry range';
            errorDiv.style.display = 'block';
            return;
        }
        const x = [...new Set(filteredData.map(item => item.Expiry))].sort((a, b) => new Date(a) - new Date(b));
        const y = [...new Set(filteredData.map(item => item.Moneyness * 100))].sort((a, b) => a - b);
        const z = [];
        y.forEach(m => {
            const row = filteredData.filter(item => item.Moneyness * 100 === m).map(item => item.Volatility);
            z.push(row);
        });
        const zValues = z.flat().filter(v => Number.isFinite(v));
        const minZ = zValues.length > 0 ? Math.min(...zValues) : 0;
        const maxZ = zValues.length > 0 ? Math.max(...zValues) : 0;
        const plotData = [{
            x: x,
            y: y,
            z: z,
            type: 'surface',
            colorscale: 'Viridis',
            cmin: minZ,
            cmax: maxZ,
            showscale: true,
            colorbar: {
                title: 'Volatility (%)',
                titleside: 'right',
                titlefont: { color: '#FFFFFF' },
                tickfont: { color: '#FFFFFF' }
            }
        }];
        const layout = {
            title: { text: 'Call Implied Volatility Surface (%)', font: { color: '#FFFFFF', size: 16 }, y: 0.95, x: 0.5, xanchor: 'center' },
            scene: {
                xaxis: { title: { text: 'Expiry', color: '#FFFFFF' }, type: 'date', tickformat: '%Y-%m-%d', color: '#FFFFFF', showgrid: false },
                yaxis: { title: { text: 'Moneyness (%)', color: '#FFFFFF' }, color: '#FFFFFF', showgrid: false },
                zaxis: { title: { text: 'Volatility (%)', color: '#FFFFFF' }, color: '#FFFFFF', range: [minZ, maxZ], showgrid: false },
                aspectmode: 'auto',
                bgcolor: '#1A1A2E'
            },
            margin: { l: 40, r: 40, b: 40, t: 60 },
            autosize: true,
            paper_bgcolor: '#1A1A2E',
            plot_bgcolor: '#1A1A2E'
        };
        Plotly.newPlot('call-vol-surface', plotData, layout);
        document.getElementById('moneyness-value').textContent = `${moneynessMin.toFixed(1)} - ${moneynessMax.toFixed(1)}`;
        document.getElementById('expiry-t-value').textContent = `${expiryTMin.toFixed(1)} - ${expiryTMax.toFixed(1)}`;
    } catch (error) {
        console.error('Error updating Call Volatility Surface:', error);
        Plotly.purge('call-vol-surface');
        document.getElementById('surface-error').textContent = `Error rendering volatility surface: ${error.message}`;
        document.getElementById('surface-error').style.display = 'block';
    }
}

function updateHistoricChart() {
    try {
        const selectedTicker = document.getElementById('ticker-search').value.toUpperCase();
        const historyLength = document.getElementById('history-select').value;
        const errorDiv = document.getElementById('historic-error');
        errorDiv.style.display = 'none';
        let filteredHistoric = historicData.filter(item => item.Ticker === selectedTicker && item.Date && Number.isFinite(item.Close));
        if (!filteredHistoric.length) {
            if (historicChart) historicChart.destroy();
            errorDiv.textContent = 'No historic data available';
            errorDiv.style.display = 'block';
            return;
        }
        filteredHistoric.sort((a, b) => new Date(a.Date) - new Date(b.Date));
        const fromDate = calculateDateRange(historyLength);
        const minHistoricDate = new Date(Math.min(...filteredHistoric.map(item => new Date(item.Date))));
        const maxHistoricDate = new Date(Math.max(...filteredHistoric.map(item => new Date(item.Date))));
        if (fromDate) {
            filteredHistoric = filteredHistoric.filter(item => new Date(item.Date) >= fromDate);
        }
        const datasets = [
            {
                label: 'Stock Price',
                data: filteredHistoric.map(item => ({ x: new Date(item.Date), y: parseFloat(item.Close) })),
                borderColor: '#FFFF00',
                backgroundColor: '#FFFF00',
                pointRadius: 1,
                fill: false,
                showLine: true,
                yAxisID: 'y'
            }
        ];
        const annotations = eventsData
            .filter(event => {
                if (!event.Start_Date || !event.End_Date) return false;
                const startDate = new Date(event.Start_Date);
                const endDate = new Date(event.End_Date);
                if (isNaN(startDate.getTime()) || isNaN(endDate.getTime())) return false;
                if (!(event.Ticker === 'ALL' || event.Ticker === selectedTicker)) return false;
                if (startDate > maxHistoricDate || endDate < minHistoricDate) return false;
                if (fromDate && endDate < fromDate) return false;
                return true;
            })
            .map((event, index) => ({
                type: 'box',
                xMin: new Date(Math.max(new Date(event.Start_Date), fromDate || minHistoricDate)),
                xMax: new Date(Math.min(new Date(event.End_Date), maxHistoricDate)),
                yMin: 'y',
                yMax: 'y',
                backgroundColor: event.Impact === 'dip' ? 'rgba(248, 113, 113, 0.2)' : 'rgba(74, 222, 128, 0.2)',
                borderColor: event.Impact === 'dip' ? 'rgba(248, 113, 113, 0.5)' : 'rgba(74, 222, 128, 0.5)',
                borderWidth: 1,
                label: {
                    content: event.Event || 'Unknown Event',
                    display: true,
                    position: 'center',
                    color: '#FFFFFF',
                    font: { size: 12 },
                    rotation: 90
                }
            }));
        if (historicChart) historicChart.destroy();
        const canvas = document.getElementById('historic-price-chart');
        if (!canvas.getContext('2d')) throw new Error('Canvas context not available');
        historicChart = new Chart(canvas.getContext('2d'), {
            type: 'line',
            data: { datasets },
            options: {
                plugins: {
                    legend: { labels: { color: '#FFFFFF' } },
                    annotation: { annotations }
                },
                scales: {
                    x: {
                        type: 'time',
                        time: { unit: 'month' },
                        title: { display: true, text: 'Date', color: '#FFFFFF' },
                        grid: { display: false },
                        ticks: { color: '#FFFFFF' }
                    },
                    y: {
                        title: { display: true, text: 'Stock Price ($)', color: '#FFFFFF' },
                        beginAtZero: false,
                        grid: { display: false },
                        ticks: { color: '#FFFFFF' }
                    }
                },
                backgroundColor: '#1A1A2E'
            }
        });
        createStatsTable(historicChart, 'historic-price-chart');
    } catch (error) {
        console.error('Error updating Historic chart:', error);
        document.getElementById('historic-error').textContent = `Error rendering historic chart: ${error.message}`;
        document.getElementById('historic-error').style.display = 'block';
    }
}

function updateVolumeChart() {
    try {
        const selectedTicker = document.getElementById('ticker-search').value.toUpperCase();
        const uniqueTickers = [...new Set(rankingTableData.map(item => item.Ticker))].filter(val => val).sort();
        const volumes = uniqueTickers.map(ticker => {
            const tickerData = rankingTableData.find(item => item.Ticker === ticker);
            return tickerData && Number.isFinite(parseFloat(tickerData['Volume'])) ? parseInt(tickerData['Volume'].replace(/,/g, '')) : 0;
        });
        const backgroundColors = uniqueTickers.map(ticker => ticker === selectedTicker ? '#FFFF00' : '#00BFFF');
        if (volumeChart) volumeChart.destroy();
        const canvas = document.getElementById('volume-chart');
        if (!canvas.getContext('2d')) throw new Error('Canvas context not available for volume-chart');
        volumeChart = new Chart(canvas.getContext('2d'), {
            type: 'bar',
            data: {
                labels: uniqueTickers,
                datasets: [{
                    label: 'Total Volume',
                    data: volumes,
                    backgroundColor: backgroundColors,
                    borderColor: backgroundColors,
                    borderWidth: 1
                }]
            },
            options: {
                plugins: { legend: { labels: { color: '#FFFFFF' } } },
                scales: {
                    x: {
                        title: { display: true, text: 'Ticker', color: '#FFFFFF' },
                        grid: { display: false },
                        ticks: { color: '#FFFFFF' }
                    },
                    y: {
                        title: { display: true, text: 'Volume', color: '#FFFFFF' },
                        beginAtZero: true,
                        grid: { display: false },
                        ticks: {
                            color: '#FFFFFF',
                            callback: value => Number(value).toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })
                        }
                    }
                },
                backgroundColor: '#1C2526'
            }
        });
        console.log('Volume chart updated for tickers:', uniqueTickers);
    } catch (error) {
        console.error('Error updating volume chart:', error);
        document.getElementById('historic-error').textContent = `Error rendering volume chart: ${error.message}`;
        document.getElementById('historic-error').style.display = 'block';
    }
}

function updateOpenInterestChart() {
    try {
        const selectedTicker = document.getElementById('ticker-search').value.toUpperCase();
        const uniqueTickers = [...new Set(rankingTableData.map(item => item.Ticker))].filter(val => val).sort();
        const openInterests = uniqueTickers.map(ticker => {
            const tickerData = rankingTableData.find(item => item.Ticker === ticker);
            return tickerData && Number.isFinite(parseFloat(tickerData['Open Interest'])) ? parseInt(tickerData['Open Interest'].replace(/,/g, '')) : 0;
        });
        const backgroundColors = uniqueTickers.map(ticker => ticker === selectedTicker ? '#FFFF00' : '#00BFFF');
        if (openInterestChart) openInterestChart.destroy();
        const canvas = document.getElementById('open-interest-chart');
        if (!canvas.getContext('2d')) throw new Error('Canvas context not available for open-interest-chart');
        openInterestChart = new Chart(canvas.getContext('2d'), {
            type: 'bar',
            data: {
                labels: uniqueTickers,
                datasets: [{
                    label: 'Total Open Interest',
                    data: openInterests,
                    backgroundColor: backgroundColors,
                    borderColor: backgroundColors,
                    borderWidth: 1
                }]
            },
            options: {
                plugins: { legend: { labels: { color: '#FFFFFF' } } },
                scales: {
                    x: {
                        title: { display: true, text: 'Ticker', color: '#FFFFFF' },
                        grid: { display: false },
                        ticks: { color: '#FFFFFF' }
                    },
                    y: {
                        title: { display: true, text: 'Open Interest', color: '#FFFFFF' },
                        beginAtZero: true,
                        grid: { display: false },
                        ticks: {
                            color: '#FFFFFF',
                            callback: value => Number(value).toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })
                        }
                    }
                },
                backgroundColor: '#1C2526'
            }
        });
        console.log('Open Interest chart updated for tickers:', uniqueTickers);
    } catch (error) {
        console.error('Error updating open interest chart:', error);
        document.getElementById('historic-error').textContent = `Error rendering open interest chart: ${error.message}`;
        document.getElementById('historic-error').style.display = 'block';
    }
}

function createStatsTable(chart, canvasId) {
    try {
        const container = document.getElementById(canvasId).parentNode;
        let existingTable = container.querySelector('.stats-table');
        if (existingTable) existingTable.remove();
        const table = document.createElement('table');
        table.className = 'stats-table';
        table.style.display = canvasId === 'historic-price-chart' ? 'table' : (statsTablesVisible ? 'table' : 'none');
        const thead = table.createTHead();
        const headerRow = thead.insertRow();
        ['Label', 'Last', 'Min', 'Max', 'Mean', 'SD', 'SD Change'].forEach(text => {
            const th = document.createElement('th');
            th.textContent = text;
            headerRow.appendChild(th);
        });
        const tbody = table.createTBody();
        chart.data.datasets.forEach(dataset => {
            if (!dataset.data.length) return;
            const ys = dataset.data.map(point => point.y).filter(y => !isNaN(y));
            if (!ys.length) return;
            const last = ys[ys.length - 1];
            const min = Math.min(...ys);
            const max = Math.max(...ys);
            const mean = ys.reduce((a, b) => a + b, 0) / ys.length;
            const sd = Math.sqrt(ys.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / (ys.length - 1)) || 0;
            const sdChanges = ys.slice(1).map((y, i) => y - ys[i]);
            const sdChange = sdChanges.length > 0 ? Math.sqrt(sdChanges.reduce((a, b) => a + Math.pow(b, 2), 0) / (sdChanges.length - 1)) || 0 : 0;
            const row = tbody.insertRow();
            const labelTd = row.insertCell();
            const swatch = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
            swatch.setAttribute('width', '20');
            swatch.setAttribute('height', '10');
            swatch.setAttribute('class', 'line-swatch');
            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', '0');
            line.setAttribute('y1', '5');
            line.setAttribute('x2', '20');
            line.setAttribute('y2', '5');
            line.setAttribute('stroke', dataset.borderColor);
            line.setAttribute('stroke-width', '2');
            if (dataset.borderDash && dataset.borderDash.length > 0) {
                line.setAttribute('stroke-dasharray', dataset.borderDash.join(' '));
            }
            swatch.appendChild(line);
            labelTd.appendChild(swatch);
            labelTd.appendChild(document.createTextNode(` ${dataset.label}`));
            row.insertCell().textContent = last.toFixed(2);
            row.insertCell().textContent = min.toFixed(2);
            row.insertCell().textContent = max.toFixed(2);
            row.insertCell().textContent = mean.toFixed(2);
            row.insertCell().textContent = sd.toFixed(2);
            row.insertCell().textContent = sdChange.toFixed(2);
        });
        container.appendChild(table);
    } catch (error) {
        console.error('Error creating stats table:', error);
        document.getElementById('historic-error').textContent = `Error creating stats table: ${error.message}`;
        document.getElementById('historic-error').style.display = 'block';
    }
}

function updateSection(sectionId) {
    try {
        console.log('Updating section:', sectionId);
        const sections = ['overview', 'volatility', 'ranking', 'stock', 'data-table'];
        sections.forEach(id => {
            const element = document.getElementById(id);
            if (id === sectionId) {
                element.style.display = id === 'overview' || id === 'volatility' ? 'grid' : 'block';
            } else {
                element.style.display = 'none';
            }
        });
        updateCharts();
    } catch (error) {
        console.error('Error updating section:', error);
        document.getElementById('historic-error').textContent = `Error navigating to section: ${error.message}`;
        document.getElementById('historic-error').style.display = 'block';
    }
}

function updateCharts() {
    try {
        console.log('Updating all charts and tables');
        updateMoneynessVsIV();
        updateExpiryVsIV();
        updateYFIVChart();
        updateSkewVsExpiry();
        updateRealisedVolChart();
        updateCallVolSurface();
        updateSummaryTable();
        updateRankingTable();
        updateStockTable();
        updateTopContractsTables();
        updateVolumeChart();
        updateOpenInterestChart();
        updateHistoricChart();
        updateRawDataTable();
    } catch (error) {
        console.error('Error updating charts:', error);
        document.getElementById('historic-error').textContent = `Error updating charts: ${error.message}`;
        document.getElementById('historic-error').style.display = 'block';
    }
}

function updateRawDataTable() {
    try {
        const table = document.getElementById('raw-data-table');
        table.innerHTML = '';
        if (!cleanedData.length) {
            const row = table.insertRow();
            const cell = row.insertCell();
            cell.textContent = 'No raw data available';
            cell.colSpan = 10;
            console.warn('No raw data for raw data table');
            return;
        }
        const thead = document.createElement('thead');
        const headerRow = thead.insertRow();
        const columns = Object.keys(cleanedData[0]);
        columns.forEach((text, index) => {
            const th = document.createElement('th');
            th.textContent = text;
            th.dataset.column = text;
            th.dataset.order = 'desc';
            th.addEventListener('click', () => sortTable('raw-data-table', index));
            headerRow.appendChild(th);
        });
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
        if (typeof Chart === 'undefined' || typeof Plotly === 'undefined' || typeof Papa === 'undefined') {
            console.error('Required libraries (Chart.js, Plotly, or PapaParse) not loaded');
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
            step: 0.1
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
                    } else {
                        console.warn('Invalid timestamp:', timestamp);
                        document.getElementById('historic-error').textContent = 'Invalid data timestamp';
                        document.getElementById('historic-error').style.display = 'block';
                    }
                }, 500));
                sourceSelect.addEventListener('change', async () => {
                    console.log('Source changed to:', sourceSelect.value);
                    const timestamp = dateSelect.value + '_' + (timeSelect.options.length > 0 ? timeSelect.value : '2136');
                    if (await validateTimestamp(timestamp)) {
                        await loadData(timestamp);
                        updateDropdowns();
                        const currentSection = document.querySelector('.overview-grid[style*="grid"], .volatility-grid[style*="grid"], .chart-container[style*="block"]');
                        const sectionId = currentSection ? currentSection.id : 'overview';
                        updateSection(sectionId);
                        console.log('Updated section after source change:', sectionId);
                    } else {
                        console.warn('Invalid timestamp:', timestamp);
                        document.getElementById('historic-error').textContent = 'Invalid data timestamp';
                        document.getElementById('historic-error').style.display = 'block';
                    }
                });
                const links = document.querySelectorAll('.sidebar a');
                links.forEach(link => {
                    link.addEventListener('click', async (e) => {
                        e.preventDefault();
                        const targetId = link.getAttribute('href').substring(1);
                        console.log(`Navigating to section: ${targetId}`);
                        const timestamp = dateSelect.value + '_' + (timeSelect.options.length > 0 ? timeSelect.value : '2136');
                        if (await validateTimestamp(timestamp)) {
                            await loadData(timestamp);
                            updateDropdowns();
                            updateSection(targetId);
                        } else {
                            console.warn('Invalid timestamp on navigation:', timestamp);
                            document.getElementById('historic-error').textContent = 'Invalid data timestamp';
                            document.getElementById('historic-error').style.display = 'block';
                        }
                    });
                });
                const overviewContainer = document.getElementById('overview');
                overviewContainer.style.display = 'grid';
                document.getElementById('volatility').style.display = 'none';
                document.getElementById('ranking').style.display = 'none';
                document.getElementById('stock').style.display = 'none';
                document.getElementById('data-table').style.display = 'none';
                console.log('Initial load: Showing Overview');
                addKeyboardNavigation();
            })
            .catch(error => {
                console.error('Error loading dates:', error);
                document.getElementById('historic-error').textContent = `Failed to load date options: ${error.message}`;
                document.getElementById('historic-error').style.display = 'block';
            });
    } catch (error) {
        console.error('Error in DOMContentLoaded:', error);
        document.getElementById('historic-error').textContent = `Error initializing page: ${error.message}`;
        document.getElementById('historic-error').style.display = 'block';
    }
});
