import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# ---- BUNDLE CONFIG EDITOR ----
st.sidebar.header("Bundle Configuration")
default_bundle_df = pd.DataFrame([
    {'bundle_key':'bundle1','name':'Bundle 1','price':35.0,'primary':1,'complementary':0,'dist_pct':54.0},
    {'bundle_key':'bundle2','name':'Bundle 2','price':69.17,'primary':2,'complementary':1,'dist_pct':35.0},
    {'bundle_key':'bundle3','name':'Bundle 3','price':109.8,'primary':4,'complementary':2,'dist_pct':11.0},
])
bundle_df = st.sidebar.data_editor(
    default_bundle_df,
    num_rows="fixed",
    use_container_width=True,
    column_config={
        'bundle_key':    st.column_config.TextColumn('Key', disabled=True),
        'name':          st.column_config.TextColumn('Name', disabled=True),
        'price':         st.column_config.NumberColumn('Price (€)', step=0.1),
        'primary':       st.column_config.NumberColumn('Primary count', step=1),
        'complementary': st.column_config.NumberColumn('Complementary count', step=1),
        'dist_pct':      st.column_config.NumberColumn('Distribution %', step=1.0),
    }
)
if abs(bundle_df['dist_pct'].sum() - 100) > 1e-6:
    st.sidebar.error("Distribution % must sum to 100")

bundle_template = {
    row.bundle_key: {
        'name': row.name,
        'price': row.price,
        'composition': {
            'primary': row.primary,
            'complementary': row.complementary
        }
    }
    for row in bundle_df.itertuples()
}
bundle_distribution = {row.bundle_key: row.dist_pct for row in bundle_df.itertuples()}

# ---- UTILITIES ----
def parse_revenue_input(s):
    s = str(s).strip().replace(',','').upper()
    if s.endswith('K'): return float(s[:-1]) * 1_000
    if s.endswith('M'): return float(s[:-1]) * 1_000_000
    try: return float(s)
    except: return 0

def format_currency(v):
    if abs(v) >= 1_000_000: return f"€{v/1_000_000:.1f}M"
    if abs(v) >= 1_000:     return f"€{v/1_000:.1f}k"
    return f"€{v:.1f}"

def format_int(v):
    try: return f"{int(round(v)):,}"
    except: return str(v)

# ---- SYNC CALLBACKS ----
def annual_changed(i):
    # Sync monthly/daily from individual lineup change
    ann = parse_revenue_input(st.session_state[f"rev_{i}"])
    st.session_state[f"mon_{i}"] = format_int(ann/12)
    st.session_state[f"day_{i}"] = format_int(ann/365)
    # Update sidebar total
    total = sum(parse_revenue_input(st.session_state.get(f"rev_{j}", 0))
                for j in range(st.session_state.num_products))
    st.session_state['annual_target'] = format_int(total)


def monthly_changed(i):
    mon = parse_revenue_input(st.session_state[f"mon_{i}"])
    ann = mon * 12
    st.session_state[f"rev_{i}"] = format_int(ann)
    st.session_state[f"day_{i}"] = format_int(mon/30)
    annual_changed(i)

def daily_changed(i):
    day = parse_revenue_input(st.session_state[f"day_{i}"])
    ann = day * 365
    st.session_state[f"rev_{i}"] = format_int(ann)
    st.session_state[f"mon_{i}"] = format_int(day*30)
    annual_changed(i)

# ---- ANNUAL TARGET CALLBACK ----
def annual_target_changed():
    ann = parse_revenue_input(st.session_state['annual_target'])
    np = st.session_state.num_products
    each = ann / np if np else 0
    for j in range(np):
        st.session_state[f"rev_{j}"] = format_int(each)
        st.session_state[f"mon_{j}"] = format_int(each/12)
        st.session_state[f"day_{j}"] = format_int(each/365)

# ---- SESSION STATE DEFAULTS ----
if 'annual_target' not in st.session_state:
    st.session_state.annual_target = '2000000'
if 'rev_per_ad' not in st.session_state:
    st.session_state.rev_per_ad = '84k'
if 'num_products' not in st.session_state:
    st.session_state.num_products = 2

# ---- SIDEBAR CAMPAIGN CONFIG ----
st.sidebar.header("Revenue & Campaign Config")
annual_target_input = st.sidebar.text_input(
    "Annual Target Revenue (€)",
    value=st.session_state.annual_target,
    key='annual_target',
    on_change=annual_target_changed
)
if not st.session_state.rev_per_ad:
    st.session_state.rev_per_ad = format_int(
        parse_revenue_input(annual_target_input)/12
    )
rev_per_ad_input = st.sidebar.text_input(
    "Revenue per Winning Ad (€/month)",
    value=st.session_state.rev_per_ad,
    key='rev_per_ad'
)
num_products = st.sidebar.selectbox(
    "# Products to Scale", [1,2,3,4,5],
    index=st.session_state.num_products-1,
    key='num_products'
)

annual_target = parse_revenue_input(st.session_state['annual_target'])
monthly_target = annual_target/12 if annual_target else 0
rev_per_ad = parse_revenue_input(st.session_state['rev_per_ad'])
num_products = st.session_state.num_products

# ---- PRODUCT LINEUP CONFIG ----
st.header("Product Lineup Configuration")
available_products = [
    'PROBIOTIC','MULTIVITA','DENTAL','SKIN AND COAT',
    'RELAX','ALLERGY','ANAL GLAND','HIP AND JOINT','UROLOGY'
]
# Defaults for lineups
default_prim = ['PROBIOTIC','HIP AND JOINT','SKIN AND COAT','DENTAL','RELAX']
default_comp = ['MULTIVITA','MULTIVITA','ALLERGY','PROBIOTIC','MULTIVITA']
# default_revs unchanged
default_revs = [2500000,1500000,1000000,1000000,1000000]

product_lineups = []
for i in range(num_products):
    with st.expander(f"Lineup {i+1}", expanded=True):
        c1, c2 = st.columns(2)
        prim_key, comp_key = f"prim_{i}", f"comp_{i}"
        rev_key, mon_key, day_key = f"rev_{i}", f"mon_{i}", f"day_{i}"
        hr_key = f"hr_{i}"

        if i < len(default_prim):
            st.session_state.setdefault(prim_key, default_prim[i])
            st.session_state.setdefault(comp_key, default_comp[i])
            st.session_state.setdefault(rev_key, format_int(default_revs[i]))
        else:
            default_ann = annual_target/num_products if annual_target else 0
            st.session_state.setdefault(prim_key, available_products[i % len(available_products)])
            st.session_state.setdefault(comp_key, available_products[i % len(available_products)])
            st.session_state.setdefault(rev_key, format_int(default_ann))

        # Sync monthly and daily
        st.session_state.setdefault(mon_key, format_int(parse_revenue_input(st.session_state[rev_key])/12))
        st.session_state.setdefault(day_key, format_int(parse_revenue_input(st.session_state[rev_key])/365))
        st.session_state.setdefault(hr_key, 15)

        primary = c1.selectbox(
            "Primary Product", available_products,
            index=available_products.index(st.session_state[prim_key]), key=prim_key
        )
        complementary = c2.selectbox(
            "Complementary Product", available_products,
            index=available_products.index(st.session_state[comp_key]), key=comp_key
        )

        st.markdown("Enter one revenue; others auto-sync:")
        r1, r2, r3 = st.columns(3)
        ann_val = parse_revenue_input(st.session_state[rev_key])
        mon_val, day_val = ann_val/12, ann_val/365

        r1.text_input("Annual Revenue (€)", format_int(ann_val), key=rev_key,
                      on_change=annual_changed, args=(i,))
        r2.text_input("Monthly Revenue (€)", format_int(mon_val), key=mon_key,
                      on_change=monthly_changed, args=(i,))
        r3.text_input("Daily Revenue (€)", format_int(day_val), key=day_key,
                      on_change=daily_changed, args=(i,))

        hit_rate = st.number_input("Hit Rate (%)", 0, 100,
                                   value=int(st.session_state[hr_key]),
                                   step=1, format="%d", key=hr_key)

        product_lineups.append({'primary': primary,
                                'complementary': complementary,
                                'annual_revenue': ann_val,
                                'hit_rate': hit_rate})

# ---- REVENUE ALLOCATION CHECK ----
total_alloc = sum(p['annual_revenue'] for p in product_lineups)
if annual_target and total_alloc != annual_target:
    diff = total_alloc - annual_target
    st.error(f"Total allocated revenue {'exceeds' if diff>0 else 'is'} {format_currency(abs(diff))} {'above' if diff>0 else 'below'} target")
else:
    st.success("Total allocated revenue matches target")

# ---- BUILD LINEUP RESULTS ----
avg_price = sum(bundle_template[k]['price'] * (bundle_distribution[k]/100) for k in bundle_template)
lineup_results = []
for ln in product_lineups:
    mrev = ln['annual_revenue']/12
    tb = int(round(mrev/avg_price))
    b1 = int(round(tb * bundle_distribution['bundle1']/100))
    b2 = int(round(tb * bundle_distribution['bundle2']/100))
    b3 = int(round(tb * bundle_distribution['bundle3']/100))
    actual = b1*bundle_template['bundle1']['price'] + b2*bundle_template['bundle2']['price'] + b3*bundle_template['bundle3']['price']
    prim_qty = (b1*bundle_template['bundle1']['composition']['primary'] +
                b2*bundle_template['bundle2']['composition']['primary'] +
                b3*bundle_template['bundle3']['composition']['primary'])
    comp_qty = (b1*bundle_template['bundle1']['composition']['complementary'] +
                b2*bundle_template['bundle2']['composition']['complementary'] +
                b3*bundle_template['bundle3']['composition']['complementary'])
    prod_qty = {ln['primary']: prim_qty + (comp_qty if ln['primary']==ln['complementary'] else 0)}
    if ln['complementary'] != ln['primary']:
        prod_qty[ln['complementary']] = comp_qty
    lineup_results.append({**ln,'monthly_revenue': mrev,'bundle_qty': {'B1':b1,'B2':b2,'B3':b3},'actual_rev': actual,'prod_qty': prod_qty,'total_bundles': tb,'total_units': sum(prod_qty.values())})

# ---- ALLOCATION SUMMARY ----
st.subheader("Allocation Summary")
cA,cB,cC = st.columns(3)
cA.metric("Annual Target", format_currency(annual_target), delta=format_currency(total_alloc-annual_target))
cB.metric("Monthly Forecast", format_currency(annual_target/12), delta=format_currency((annual_target/12)-monthly_target))
cC.metric("Allocation %", f"{int(round((total_alloc/annual_target if annual_target else 0)*100))}%")
st.progress(min((total_alloc/annual_target if annual_target else 0),1.0))

# ---- MAIN TABS ----
tab1, tab2, tab3 = st.tabs(["Ad Sets Analysis","Forecast Summary","Detailed Forecasts"])

with tab1:
    winners = sum(r['monthly_revenue']/rev_per_ad for r in lineup_results)
    sets_needed = sum(np.ceil((r['monthly_revenue']/rev_per_ad)/(r['hit_rate']/100)) for r in lineup_results)
    c1,c2,c3 = st.columns(3)
    c1.metric("Revenue/Winner", format_currency(rev_per_ad))
    c2.metric("Winners Needed", format_int(np.ceil(winners)))
    c3.metric("Ad Sets to Launch", format_int(np.ceil(sets_needed)))

    df_ads = pd.DataFrame([{ 'Product': f"{r['primary']} + {r['complementary']}", 'Monthly Revenue': r['monthly_revenue'], 'Hit Rate (%)': r['hit_rate'], 'Winners Needed': np.ceil(r['monthly_revenue']/rev_per_ad), 'Ad Sets to Launch': np.ceil((r['monthly_revenue']/rev_per_ad)/(r['hit_rate']/100)) } for r in lineup_results])
    totals = {'Product':'TOTAL','Monthly Revenue': df_ads['Monthly Revenue'].sum(),'Hit Rate (%)':'','Winners Needed': df_ads['Winners Needed'].sum(),'Ad Sets to Launch': df_ads['Ad Sets to Launch'].sum()}
    df_ads = pd.concat([df_ads, pd.DataFrame([totals])], ignore_index=True)
    st.dataframe(df_ads.style.format({'Monthly Revenue': lambda v: format_currency(v),'Winners Needed': lambda v: format_int(v),'Ad Sets to Launch': lambda v: format_int(v),'Hit Rate (%)': lambda v: f"{int(v)}%" if v!='' else ''}), use_container_width=True)

with tab2:
    m1,m2,m3 = st.columns(3)
    m1.metric("Monthly Revenue", format_currency(sum(r['actual_rev'] for r in lineup_results)), f"Annual: {format_currency(sum(r['actual_rev'] for r in lineup_results)*12)}")
    total_b1 = sum(r['bundle_qty']['B1'] for r in lineup_results)
    total_b2 = sum(r['bundle_qty']['B2'] for r in lineup_results)
    total_b3 = sum(r['bundle_qty']['B3'] for r in lineup_results)
    m2.metric("Total Bundles", format_int(total_b1+total_b2+total_b3), f"B1:{format_int(total_b1)} | B2:{format_int(total_b2)} | B3:{format_int(total_b3)}")
    combined_prod = {}
    for r in lineup_results:
        for p,q in r['prod_qty'].items(): combined_prod[p] = combined_prod.get(p,0) + q
    m3.metric("Total Units", format_int(sum(combined_prod.values())))

    # Pie chart removed per request

    prod_df = pd.DataFrame({'Product':list(combined_prod.keys()), 'Units':list(combined_prod.values())}).nlargest(5,'Units')
    bar = alt.Chart(prod_df).mark_bar().encode(x='Product:N', y='Units:Q', tooltip=['Product','Units'])
    st.altair_chart(bar, use_container_width=True)

    df_req = pd.DataFrame([{'Product':p,'Monthly Units':q,'Annual Units':q*12} for p,q in combined_prod.items()])
    st.table(df_req.assign(**{'Monthly Units': df_req['Monthly Units'].map(format_int),'Annual Units': df_req['Annual Units'].map(format_int)}))

with tab3:
    for idx, r in enumerate(lineup_results, start=1):
        st.subheader(f"Lineup {idx}: {r['primary']} + {r['complementary']}")
        prim_name, comp_name = r['primary'], r['complementary']
        same = prim_name == comp_name
        rows = []
        for key, label in zip(['bundle1','bundle2','bundle3'], ['B1','B2','B3']):
            qty = r['bundle_qty'][label]
            price = bundle_template[key]['price']
            rev = qty * price
            pu = qty * bundle_template[key]['composition']['primary']
            cu = qty * bundle_template[key]['composition']['complementary']
            entry = {'Bundle':label,'Qty':qty,'Price':price,'Revenue':rev}
            if same:
                entry[f"{prim_name} Units"] = pu + cu
            else:
                entry[f"{prim_name} Units"] = pu
                entry[f"{comp_name} Units"] = cu
            rows.append(entry)
        df_rev = pd.DataFrame(rows)
        df_rev['% of Revenue'] = df_rev['Revenue'].div(df_rev['Revenue'].sum())*100
        totals = {'Bundle':'TOTAL','Qty':df_rev['Qty'].sum(),'Price':'','Revenue':df_rev['Revenue'].sum(),'% of Revenue':''}
        totals[f"{prim_name} Units"] = df_rev[f"{prim_name} Units"].sum()
        if not same:
            totals[f"{comp_name} Units"] = df_rev[f"{comp_name} Units"].sum()
        df_rev = pd.concat([df_rev,pd.DataFrame([totals])], ignore_index=True)
        fmt_map = {'Qty': lambda v: format_int(v),'Price': lambda v: format_currency(v) if v!='' else '','Revenue': lambda v: format_currency(v),'% of Revenue': lambda v: f"{v:.1f}%" if v!='' else ''}
        fmt_map[f"{prim_name} Units"] = lambda v: format_int(v)
        if not same: fmt_map[f"{comp_name} Units"] = lambda v: format_int(v)
        st.dataframe(df_rev.style.format(fmt_map), use_container_width=True)

# ---- BUNDLE STRUCTURE INFO ----
st.header("Bundle Structure Info")
info_df = pd.DataFrame([{ 'Bundle': bundle_template[k]['name'], 'Price': bundle_template[k]['price'], 'Composition': f"{bundle_template[k]['composition']['primary']}P, {bundle_template[k]['composition']['complementary']}C", 'Distribution (%)': bundle_distribution[k] } for k in bundle_template])
st.table(info_df.assign(**{'Price': info_df['Price'].map(format_currency),'Distribution (%)': info_df['Distribution (%)'].map(lambda x: f"{x:.1f}%")}))
