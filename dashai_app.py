import io
import re
import zipfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title='DashAI | Business Intelligence', page_icon='📊', layout='wide', initial_sidebar_state='expanded')

# ---------- Core data interpretation ----------
MEASURE_HINTS = {
    'revenue': ['revenue', 'sales', 'amount', 'turnover', 'income', 'net sales', 'total sales'],
    'cost': ['cost', 'expense', 'expenses', 'total exp', 'cogs', 'shipping cost', 'purchase cost'],
    'profit': ['profit', 'gross profit', 'net profit', 'net inc', 'net income'],
    'quantity': ['quantity', 'qty', 'units', 'volume'],
}
DIMENSION_HINTS = {
    'date': ['date', 'day', 'month', 'year', 'time'],
    'product': ['product', 'item', 'sku', 'category'],
    'customer': ['customer', 'client', 'account'],
    'region': ['region', 'country', 'city', 'state', 'area', 'location'],
}


def clean_name(x):
    return re.sub(r'[^a-z0-9]+', ' ', str(x).lower()).strip()


def semantic_role(col, series):
    name = clean_name(col)
    n = len(series)
    unique = series.nunique(dropna=True)
    if pd.api.types.is_datetime64_any_dtype(series):
        return 'Date / Time', 'date'
    if any(k in name.split() for k in ['id', 'code']):
        return 'Identifier', 'identifier'
    # Try to recognise date-looking text columns too.
    if series.dtype == 'object' and unique > 0:
        sample = series.dropna().astype(str).head(30)
        parsed = pd.to_datetime(sample, errors='coerce')
        if len(sample) >= 5 and parsed.notna().mean() >= 0.8:
            return 'Date / Time', 'date'
    if pd.api.types.is_numeric_dtype(series):
        for semantic, hints in MEASURE_HINTS.items():
            if any(h in name for h in hints):
                return 'Measure', semantic
        return 'Measure', 'numeric measure'
    if unique <= min(30, max(5, int(n * .2))):
        for semantic, hints in DIMENSION_HINTS.items():
            if any(h in name for h in hints):
                return 'Category / Dimension', semantic
        return 'Category / Dimension', 'category'
    return 'Text / Dimension', 'text dimension'


def profile_data(df):
    rows = []
    for col in df.columns:
        s = df[col]
        role, semantic = semantic_role(col, s)
        rows.append({
            'Column': str(col),
            'Data type': str(s.dtype),
            'Suggested role': role,
            'Business meaning': semantic,
            'Unique values': int(s.nunique(dropna=True)),
            'Missing values': int(s.isna().sum()),
        })
    return pd.DataFrame(rows)


def detect_issues(df):
    issues = []
    for col in df.columns:
        count = int(df[col].isna().sum())
        if count:
            issues.append({'Issue': 'Missing values', 'Column': str(col), 'Count': count, 'Suggested action': 'Review or fill'})
    duplicates = int(df.duplicated().sum())
    if duplicates:
        issues.append({'Issue': 'Duplicate rows', 'Column': 'Entire row', 'Count': duplicates, 'Suggested action': 'Review or remove'})
    return pd.DataFrame(issues)


def find_col(df, semantic):
    for col in df.columns:
        _, s = semantic_role(col, df[col])
        if s == semantic:
            return col
    return None


def parse_date_column(df, col):
    if not col:
        return df
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        converted = pd.to_datetime(df[col], errors='coerce')
        if converted.notna().sum() > 0:
            df = df.copy()
            df[col] = converted
    return df



def normalize_financial_matrix(df):
    """Convert a month-across-columns financial report into one row per month."""
    if df.shape[1] < 3:
        return df

    month_names = {
        'jan': 1, 'january': 1, 'feb': 2, 'february': 2,
        'mar': 3, 'march': 3, 'apr': 4, 'april': 4,
        'may': 5, 'jun': 6, 'june': 6, 'jul': 7, 'july': 7,
        'aug': 8, 'august': 8, 'sep': 9, 'sept': 9, 'september': 9,
        'oct': 10, 'october': 10, 'nov': 11, 'november': 11,
        'dec': 12, 'december': 12,
    }

    month_row = None
    month_cols = []
    for ridx in range(min(len(df), 15)):
        candidate = []
        for cidx in range(1, df.shape[1]):
            value = str(df.iloc[ridx, cidx]).strip().lower()
            if value in month_names:
                candidate.append((cidx, month_names[value]))
        if len(candidate) >= 3:
            month_row, month_cols = ridx, candidate
            break

    if month_row is None:
        return df

    # Find the report year from a date-like value near the top; default to current year only
    # when the workbook does not provide one.
    year = None
    for ridx in range(min(month_row + 1, len(df))):
        for value in df.iloc[ridx].tolist():
            parsed = pd.to_datetime(value, errors='coerce')
            if pd.notna(parsed):
                year = int(parsed.year)
                break
        if year:
            break
    if year is None:
        year = pd.Timestamp.today().year

    result = {}
    for ridx in range(month_row + 1, len(df)):
        label = df.iloc[ridx, 0]
        if pd.isna(label):
            continue
        label = str(label).strip()
        if not label or label.lower() in month_names:
            continue
        values = {}
        for cidx, month_number in month_cols:
            value = pd.to_numeric(df.iloc[ridx, cidx], errors='coerce')
            if pd.notna(value):
                values[month_number] = value
        if values:
            normalized_label = label.strip()
            key = normalized_label.lower()
            if key.startswith('total exp'):
                normalized_label = 'Total Expenses'
            elif key.startswith('net inc'):
                normalized_label = 'Net Income'
            result[normalized_label] = values

    if not result:
        return df

    month_numbers = sorted({m for vals in result.values() for m in vals})
    rows = []
    for month_number in month_numbers:
        row = {'Date': pd.Timestamp(year=year, month=month_number, day=1)}
        for label, values in result.items():
            row[label] = values.get(month_number, pd.NA)
        rows.append(row)
    normalized = pd.DataFrame(rows)
    for col in normalized.columns[1:]:
        normalized[col] = pd.to_numeric(normalized[col], errors='coerce')
    return normalized


def build_plan(df, purpose, focus):
    revenue = find_col(df, 'revenue')
    cost = find_col(df, 'cost')
    profit = find_col(df, 'profit')
    quantity = find_col(df, 'quantity')
    date = find_col(df, 'date')
    product = find_col(df, 'product')
    customer = find_col(df, 'customer')
    region = find_col(df, 'region')
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    measures = []
    if revenue:
        measures.append(('Revenue', revenue))
    if profit:
        measures.append(('Profit', profit))
    if quantity:
        measures.append(('Quantity', quantity))
    if cost:
        measures.append(('Cost', cost))

    derived = []
    if revenue and cost:
        derived = ['Profit = Revenue − Cost', 'Profit Margin = Profit ÷ Revenue']

    kpis = []
    if revenue:
        kpis.append('Total Revenue')
    if profit or (revenue and cost):
        kpis.append('Total Profit')
    if revenue and date:
        kpis.append('Revenue Growth')
    if quantity:
        kpis.append('Total Quantity')
    if customer:
        kpis.append('Customer Count')
    if not kpis:
        kpis = [f'Total {c}' for c in numeric[:3]]
    kpis = kpis[:4]

    charts = []
    def add(title, chart_type, x=None, y=None, agg='sum', limit=None, color=None, derived_metric=None):
        charts.append({'title': title, 'type': chart_type, 'x': x, 'y': y, 'agg': agg,
                       'limit': limit, 'color': color, 'derived': derived_metric})

    if date and revenue:
        add(f'{revenue} over time', 'Line', date, revenue, 'sum')
    if product and revenue:
        add(f'{revenue} by {product}', 'Bar', product, revenue, 'sum')
    if region and revenue:
        add(f'{revenue} by {region}', 'Bar', region, revenue, 'sum')
    if customer and revenue:
        add(f'Top customers by {revenue}', 'Bar', customer, revenue, 'sum', 10)

    if (('Profitability' in focus) or purpose == 'Finance') and (profit or (revenue and cost)):
        if date:
            add('Profit over time', 'Line', date, profit or '__derived_profit__', 'sum', derived_metric='profit')
        elif product and revenue:
            add('Profit by product', 'Bar', product, profit or '__derived_profit__', 'sum', derived_metric='profit' if not profit else None)

    if 'Products' in focus and product and revenue:
        add('Top products by revenue', 'Bar', product, revenue, 'sum', 10)
    if 'Regions' in focus and region and revenue:
        add('Regional revenue performance', 'Bar', region, revenue, 'sum')
    if 'Trends' in focus and date and quantity:
        add('Quantity over time', 'Line', date, quantity, 'sum')

    if not charts and len(numeric) >= 2:
        add(f'{numeric[1]} vs {numeric[0]}', 'Bar', numeric[0], numeric[1], 'sum')

    # Deduplicate by title while preserving order.
    seen = set()
    clean = []
    for c in charts:
        if c['title'] not in seen:
            clean.append(c)
            seen.add(c['title'])
    return {'kpis': kpis, 'charts': clean[:8], 'derived': derived, 'profile': profile_data(df)}


def add_derived_columns(df, enabled):
    df = df.copy()
    revenue = find_col(df, 'revenue')
    cost = find_col(df, 'cost')
    profit = find_col(df, 'profit')
    if 'Profit = Revenue − Cost' in enabled and revenue and cost:
        df['__DashAI_Profit'] = df[revenue] - df[cost]
    if 'Profit Margin = Profit ÷ Revenue' in enabled:
        profit_col = profit or ('__DashAI_Profit' if '__DashAI_Profit' in df else None)
        if profit_col and revenue:
            df['__DashAI_ProfitMargin'] = df[profit_col] / df[revenue].replace(0, pd.NA)
    return df


def metric_value(df, label):
    revenue = find_col(df, 'revenue')
    cost = find_col(df, 'cost')
    profit = find_col(df, 'profit')
    quantity = find_col(df, 'quantity')
    customer = find_col(df, 'customer')
    if label == 'Total Revenue' and revenue:
        return df[revenue].sum()
    if label == 'Total Profit':
        col = profit or ('__DashAI_Profit' if '__DashAI_Profit' in df else None)
        if col:
            return df[col].sum()
        if revenue and cost:
            return (df[revenue] - df[cost]).sum()
    if label == 'Total Quantity' and quantity:
        return df[quantity].sum()
    if label == 'Customer Count' and customer:
        return df[customer].nunique(dropna=True)
    if label == 'Revenue Growth' and revenue:
        date = find_col(df, 'date')
        if date and len(df) > 1:
            tmp = df[[date, revenue]].dropna().sort_values(date)
            if len(tmp) >= 2:
                g = tmp.groupby(tmp[date].dt.to_period('M'))[revenue].sum()
                if len(g) >= 2 and g.iloc[-2] != 0:
                    return (g.iloc[-1] / g.iloc[-2]) - 1
    if label.startswith('Total '):
        col = label[6:]
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            return df[col].sum()
    return None


def format_metric(label, value):
    if value is None or pd.isna(value):
        return '—'
    if 'Growth' in label or 'Margin' in label:
        return f'{value:.1%}'
    if isinstance(value, (int, float)):
        return f'{value:,.2f}' if abs(value) < 100 else f'{value:,.0f}'
    return str(value)


def render_chart(df, spec, chart_type):
    work = df.copy()
    x, y = spec.get('x'), spec.get('y')
    if spec.get('derived') == 'profit':
        y = '__DashAI_Profit' if '__DashAI_Profit' in work.columns else y
    if not x or not y or x not in work.columns or y not in work.columns:
        st.warning(f"Not enough usable data to render **{spec['title']}**.")
        return

    work = work[[x, y]].dropna()
    if work.empty:
        st.warning(f"No usable rows remain for **{spec['title']}** after filtering.")
        return
    if pd.api.types.is_datetime64_any_dtype(work[x]):
        work = work.sort_values(x)

    if spec.get('limit') and chart_type in ('Bar', 'Horizontal Bar'):
        grouped = work.groupby(x, dropna=False)[y].sum().reset_index().sort_values(y, ascending=False).head(spec['limit'])
    elif chart_type in ('Bar', 'Horizontal Bar'):
        grouped = work.groupby(x, dropna=False)[y].sum().reset_index()
        if len(grouped) > 30:
            grouped = grouped.sort_values(y, ascending=False).head(30)
    elif chart_type == 'Line':
        if pd.api.types.is_datetime64_any_dtype(work[x]):
            grouped = work.groupby(x, dropna=False)[y].sum().reset_index().sort_values(x)
        else:
            grouped = work.groupby(x, dropna=False)[y].sum().reset_index()
    else:
        grouped = work.groupby(x, dropna=False)[y].sum().reset_index()

    label_y = y
    if y == '__DashAI_Profit':
        label_y = 'Profit'
    grouped = grouped.rename(columns={y: label_y})

    if chart_type == 'Line':
        fig = px.line(grouped, x=x, y=label_y, markers=True, title=spec['title'])
    elif chart_type == 'Horizontal Bar':
        fig = px.bar(grouped, x=label_y, y=x, orientation='h', title=spec['title'])
    elif chart_type == 'Pie':
        pie = grouped.head(12)
        fig = px.pie(pie, names=x, values=label_y, title=spec['title'])
    elif chart_type == 'Scatter':
        fig = px.scatter(work, x=x, y=y, title=spec['title'])
    else:
        fig = px.bar(grouped, x=x, y=label_y, title=spec['title'])
    fig.update_layout(
        template='plotly_white',
        colorway=['#3563E9', '#14B8A6', '#F59E0B', '#8B5CF6', '#EC4899'],
        font=dict(family='Arial, sans-serif', color='#263247', size=12),
        title=dict(font=dict(size=16, color='#172033'), x=0.02, xanchor='left'),
        height=390,
        margin=dict(l=24, r=24, t=58, b=24),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hoverlabel=dict(bgcolor='#172033', font_color='white'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------- UI ----------
st.caption('BUSINESS INTELLIGENCE WORKSPACE')
st.title('DashAI')
st.markdown('### Turn your data into decisions.')
st.caption('Upload a spreadsheet, understand your data, and build an interactive business dashboard — in minutes.')
st.divider()

st.subheader('Start with your data')
st.write('Upload an Excel or CSV file to begin your analysis.')
uploaded = st.file_uploader('Choose a dataset', type=['xlsx', 'xls', 'csv'], help='Supported formats: .xlsx, .xls and .csv')

if not uploaded:
    st.info('Your workspace is ready. Upload a dataset above to get started.')
    st.stop()

try:
    if uploaded.name.lower().endswith('.csv'):
        df = pd.read_csv(uploaded)
        sheet_name = 'CSV'
    else:
        raw = uploaded.getvalue()
        xls = pd.ExcelFile(io.BytesIO(raw))
        sheet_name = st.selectbox('Choose a sheet', xls.sheet_names)
        df = pd.read_excel(io.BytesIO(raw), sheet_name=sheet_name)

    if not uploaded.name.lower().endswith('.csv'):
        df = normalize_financial_matrix(df)

    df = df.dropna(axis=0, how='all').dropna(axis=1, how='all')
    df = parse_date_column(df, find_col(df, 'date'))

    st.success(f'Loaded **{len(df):,} rows** and **{len(df.columns):,} columns** from **{sheet_name}**.')

    st.header('01  ·  Data overview')
    st.caption('A quick profile of the structure and quality of your dataset.')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Rows', f'{len(df):,}')
    c2.metric('Columns', f'{len(df.columns):,}')
    c3.metric('Missing cells', f'{int(df.isna().sum().sum()):,}')
    c4.metric('Duplicate rows', f'{int(df.duplicated().sum()):,}')

    with st.expander('Data preview', expanded=True):
        st.dataframe(df.head(20), use_container_width=True, hide_index=True)
    with st.expander('Detected fields', expanded=True):
        st.dataframe(profile_data(df), use_container_width=True, hide_index=True)

    st.header('02  ·  Data quality')
    st.caption('Review missing values and duplicate records before building your dashboard.')
    issues = detect_issues(df)
    issue_mode = 'none'
    if not issues.empty:
        st.warning(f'{len(issues)} issue type(s) detected.')
        st.dataframe(issues, use_container_width=True, hide_index=True)
        action = st.radio(
            'What would you like to do with these data issues?',
            ['Fix automatically', 'Edit myself', 'Ignore for this dashboard'],
            horizontal=True,
        )
        issue_mode = action
        if action == 'Fix automatically':
            st.info('DashAI only applies safe fixes automatically. It does not invent missing values.')
            if st.button('Apply safe automatic fixes'):
                before = len(df)
                df = df.drop_duplicates().copy()
                st.success(f'Removed {before - len(df):,} exact duplicate row(s). Missing values remain flagged for review.')
        elif action == 'Edit myself':
            df = st.data_editor(df, use_container_width=True, num_rows='dynamic')
        else:
            st.info('The original data stays unchanged. Rows with missing values will be excluded from calculations only where the affected field is required.')
    else:
        st.success('No basic missing-value or duplicate-row issues detected.')

    st.header('03  ·  Dashboard preferences')
    st.caption('Tell DashAI what this dashboard should help you understand.')
    purpose = st.radio(
        'What is the main purpose of this dashboard?',
        ['Management', 'Sales', 'Marketing', 'Finance', 'Operations', 'Customer Analytics', 'Other'],
        horizontal=True,
    )
    focus = st.multiselect(
        'What would you like to focus on? (Select up to 3)',
        ['Sales / Revenue', 'Profitability', 'Customers', 'Products', 'Trends', 'Regions', 'Performance', 'Other'],
        max_selections=3,
    )

    st.header('04  ·  Dashboard builder')
    st.caption('Review the recommended KPIs and visuals, then customise your selection.')
    plan = build_plan(df, purpose, focus)
    st.write(f'**Purpose:** {purpose}')
    st.write(f'**Focus:** {", ".join(focus) if focus else "AI recommends based on the dataset"}')

    st.subheader('Recommended KPIs')
    st.caption('Tick the KPIs you want to include in your dashboard.')
    selected_kpis = []
    for i, kpi in enumerate(plan['kpis']):
        if st.checkbox(kpi, value=True, key=f'kpi_select_{i}_{clean_name(kpi)}'):
            selected_kpis.append(kpi)

    enabled_derived = []
    if plan['derived']:
        st.subheader('Calculated metrics')
        st.caption('Choose which calculated fields DashAI should create from your data.')
        for i, item in enumerate(plan['derived']):
            if st.checkbox(item, value=True, key=f'derived_select_{i}_{clean_name(item)}'):
                enabled_derived.append(item)

    st.subheader('Recommended visuals')
    st.caption('Tick each chart you want to include. You can change its chart type and fields below.')
    selected_specs = []
    for i, spec in enumerate(plan['charts']):
        if st.checkbox(spec['title'], value=True, key=f'visual_select_{i}_{clean_name(spec["title"])}'):
            selected_specs.append(spec.copy())

    st.subheader('Visual editor')
    chart_type_options = ['Bar', 'Line', 'Horizontal Bar', 'Pie', 'Scatter']
    edited_specs = []
    for i, spec in enumerate(selected_specs):
        with st.expander(f"✏️ {spec['title']}", expanded=False):
            c1, c2 = st.columns(2)
            new_type = c1.selectbox('Chart type', chart_type_options, index=chart_type_options.index(spec['type']) if spec['type'] in chart_type_options else 0, key=f'chart_type_{i}')
            available_cols = list(df.columns)
            if spec.get('x') in available_cols:
                new_x = c2.selectbox('X / category', available_cols, index=available_cols.index(spec['x']), key=f'chart_x_{i}')
            else:
                new_x = c2.selectbox('X / category', available_cols, key=f'chart_x_{i}')
            y_options = [c for c in available_cols if pd.api.types.is_numeric_dtype(df[c])] + ['__DashAI_Profit'] if '__DashAI_Profit' in df.columns else [c for c in available_cols if pd.api.types.is_numeric_dtype(df[c])]
            if not y_options:
                y_options = available_cols
            y_default = spec.get('y') if spec.get('y') in y_options else y_options[0]
            new_y = st.selectbox('Y / measure', y_options, index=y_options.index(y_default), key=f'chart_y_{i}')
            edited = spec.copy()
            edited['type'], edited['x'], edited['y'] = new_type, new_x, new_y
            edited_specs.append(edited)

    if st.button('Generate Dashboard', type='primary'):
        if not edited_specs and not selected_kpis:
            st.warning('Select at least one KPI or one chart before generating the dashboard.')
        else:
            st.session_state['generated_plan'] = {
                'kpis': selected_kpis.copy(),
                'charts': [spec.copy() for spec in edited_specs],
                'derived': enabled_derived.copy(),
                'purpose': purpose,
                'focus': focus.copy(),
            }
            st.success('Your dashboard has been generated below.')

    if 'generated_plan' in st.session_state:
        gp = st.session_state['generated_plan']
        work_df = add_derived_columns(df, gp.get('derived', []))
        date_col = find_col(work_df, 'date')
        work_df = parse_date_column(work_df, date_col)

        st.divider()
        st.header('05  ·  Your interactive dashboard')
        st.caption('Explore your results. Filters and chart choices update the dashboard without changing your source file.')

        # Dashboard filters
        filter_cols = [c for c in [date_col, find_col(work_df, 'product'), find_col(work_df, 'region')] if c]
        if filter_cols:
            st.subheader('Filters')
            fcols = st.columns(len(filter_cols))
            filtered = work_df.copy()
            for i, col in enumerate(filter_cols):
                if col == date_col:
                    valid = filtered[col].dropna()
                    if not valid.empty:
                        start, end = fcols[i].date_input('Date range', value=(valid.min().date(), valid.max().date()), key='dash_date')
                        filtered = filtered[(filtered[col].dt.date >= start) & (filtered[col].dt.date <= end)]
                else:
                    vals = sorted(filtered[col].dropna().astype(str).unique().tolist())
                    chosen = fcols[i].multiselect(col, vals, default=[], key='filter_' + str(col))
                    if chosen:
                        filtered = filtered[filtered[col].astype(str).isin(chosen)]
        else:
            filtered = work_df.copy()

        st.caption(f'Showing {len(filtered):,} of {len(work_df):,} rows after filters.')

        # KPI cards
        if gp['kpis']:
            cols = st.columns(min(4, len(gp['kpis'])))
            for i, kpi in enumerate(gp['kpis'][:4]):
                value = metric_value(filtered, kpi)
                cols[i % len(cols)].metric(kpi, format_metric(kpi, value))

        # Charts
        if gp['charts']:
            for i in range(0, len(gp['charts']), 2):
                cols = st.columns(2)
                for j, spec in enumerate(gp['charts'][i:i+2]):
                    with cols[j]:
                        render_chart(filtered, spec, spec.get('type', 'Bar'))
        elif not gp['kpis']:
            st.info('No charts or KPIs were selected. Return to Step 4 and select at least one item.')
        else:
            st.info('This dashboard contains KPIs only. Select one or more visuals in Step 4 to add charts.')

        with st.expander('Dashboard configuration', expanded=False):
            st.json({'purpose': gp['purpose'], 'focus': gp['focus'], 'kpis': gp['kpis'], 'charts': gp['charts'], 'derived': gp['derived']})

        st.caption('DashAI · Interactive Business Intelligence')

except Exception as e:
    st.error(f'Could not read this file: {e}')
