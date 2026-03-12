import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="Plus Silver EDF Scenario Model",
    page_icon="📊",
    layout="wide",
)

DATA_DIR = Path(__file__).parent

# ── Data loading (cached) ────────────────────────────────────────────────────

@st.cache_data
def load_data():
    silver = pd.read_csv(DATA_DIR / "edf_silver_data.csv")
    gold = pd.read_csv(DATA_DIR / "edf_gold_payg_data.csv")
    payg = pd.read_csv(DATA_DIR / "edf_payg_bucketed.csv")
    partner = pd.read_csv(DATA_DIR / "edf_partner_payg_bucketed.csv")

    df = silver.merge(
        gold[["ZONE_CODE", "GOLD_ORDERS", "AVG_GOLD_SF",
              "AVG_GOLD_DF_NO_EDF", "AVG_GOLD_FEES_NO_EDF"]],
        on="ZONE_CODE", how="left",
    )
    for col in ["GOLD_ORDERS", "AVG_GOLD_SF", "AVG_GOLD_DF_NO_EDF", "AVG_GOLD_FEES_NO_EDF"]:
        df[col] = df[col].fillna(0)

    return df, payg, partner


@st.cache_data
def build_payg_lookup(payg_df, count_col, pct_cols):
    BUCKET_BOUNDARIES = [0, 1.5, 2.5, 3.5, 5.0, 7.0]
    BUCKET_LABELS = ["0-1.5", "1.5-2.5", "2.5-3.5", "3.5-5", "5-7", "7+"]
    EDF_THRESHOLDS = [0.99, 1.49, 1.99, 2.49, 2.99, 3.99, 4.99]

    result = {}
    for zc, grp in payg_df.groupby("ZONE_CODE"):
        bd = {r["DIST_BUCKET"]: r for _, r in grp.iterrows()}
        cum = {}
        for bi in range(len(BUCKET_LABELS)):
            bk = BUCKET_BOUNDARIES[bi]
            tot = 0
            wp = {t: 0.0 for t in EDF_THRESHOLDS}
            for bj in range(bi, len(BUCKET_LABELS)):
                r = bd.get(BUCKET_LABELS[bj])
                if r is not None:
                    n = float(r[count_col])
                    tot += n
                    for k, c in zip(EDF_THRESHOLDS, pct_cols):
                        wp[k] += n * float(r[c]) / 100.0
            cum[bk] = {
                "total": tot,
                **{k: wp[k] / tot if tot else 0.0 for k in EDF_THRESHOLDS},
            }
        result[zc] = cum
    return result


# ── Computation engine ────────────────────────────────────────────────────────

BUCKET_BOUNDARIES = [0, 1.5, 2.5, 3.5, 5.0, 7.0]
EDF_THRESHOLDS = [0.99, 1.49, 1.99, 2.49, 2.99, 3.99, 4.99]


def _interp(row, cols, vals, target):
    if target <= vals[0]:
        return row[cols[0]]
    if target >= vals[-1]:
        return row[cols[-1]]
    for i in range(len(vals) - 1):
        if vals[i] <= target <= vals[i + 1]:
            f = (target - vals[i]) / (vals[i + 1] - vals[i])
            return row[cols[i]] + f * (row[cols[i + 1]] - row[cols[i]])
    return row[cols[-1]]


def _payg_frac(cum_dict, zc, bkm, edf_amt):
    if zc not in cum_dict:
        return 0.0
    cum = cum_dict[zc]
    bb = BUCKET_BOUNDARIES[0]
    for b in BUCKET_BOUNDARIES:
        if b <= bkm:
            bb = b
        else:
            break
    pcts = cum.get(bb, {k: 0.0 for k in EDF_THRESHOLDS})
    if edf_amt <= EDF_THRESHOLDS[0]:
        return pcts[EDF_THRESHOLDS[0]]
    if edf_amt >= EDF_THRESHOLDS[-1]:
        return pcts[EDF_THRESHOLDS[-1]]
    for i in range(len(EDF_THRESHOLDS) - 1):
        if EDF_THRESHOLDS[i] <= edf_amt <= EDF_THRESHOLDS[i + 1]:
            f = (edf_amt - EDF_THRESHOLDS[i]) / (EDF_THRESHOLDS[i + 1] - EDF_THRESHOLDS[i])
            return pcts[EDF_THRESHOLDS[i]] + f * (pcts[EDF_THRESHOLDS[i + 1]] - pcts[EDF_THRESHOLDS[i]])
    return pcts[EDF_THRESHOLDS[-1]]


def _est_above(row, threshold_km, pctl_cols, pctl_values):
    for i in range(len(pctl_cols) - 1, -1, -1):
        if row[pctl_cols[i]] <= threshold_km:
            p_below = pctl_values[i]
            if i < len(pctl_cols) - 1:
                d_lo = row[pctl_cols[i]]
                d_hi = row[pctl_cols[i + 1]]
                if d_hi > d_lo:
                    f = (threshold_km - d_lo) / (d_hi - d_lo)
                    p_at = p_below + f * (pctl_values[i + 1] - p_below)
                else:
                    p_at = p_below
                return max(0, (100 - p_at) / 100.0)
            else:
                return max(0, (100 - p_below) / 100.0)
    return 1.0


def run_scenario(df, payg_cum, partner_cum, edf_amount, num_tiers,
                 tier_increment, order_threshold_pct, selection_constraint_pct):
    o_pctl_cols = [c for c in df.columns if c.startswith("ORDER_P")]
    o_pctl_vals = [int(c.replace("ORDER_P", "")) for c in o_pctl_cols]
    p_pctl_cols = [c for c in df.columns if c.startswith("PARTNER_P")]
    p_pctl_vals = [int(c.replace("PARTNER_P", "")) for c in p_pctl_cols]

    order_pctl = 100 - order_threshold_pct
    partner_pctl = 100 - selection_constraint_pct

    results = []
    for _, row in df.iterrows():
        o_km = _interp(row, o_pctl_cols, o_pctl_vals, order_pctl)
        p_km = _interp(row, p_pctl_cols, p_pctl_vals, partner_pctl)
        b_km = max(o_km, p_km)
        constraint = "Selection" if p_km > o_km else "Order"
        pct_o = _est_above(row, b_km, o_pctl_cols, o_pctl_vals)
        pct_p = _est_above(row, b_km, p_pctl_cols, p_pctl_vals)
        raw_edf = row["SILVER_ORDERS"] * pct_o
        tiers = [edf_amount + i * tier_increment for i in range(num_tiers)]
        per_tier = raw_edf / num_tiers
        rev = 0
        con_orders = 0
        for t in tiers:
            pf = _payg_frac(payg_cum, row["ZONE_CODE"], b_km, t)
            c = per_tier * pf
            con_orders += c
            rev += c * t

        ppf = _payg_frac(partner_cum, row["ZONE_CODE"], b_km, edf_amount)
        bb = BUCKET_BOUNDARIES[0]
        for b in BUCKET_BOUNDARIES:
            if b <= b_km:
                bb = b
            else:
                break
        part_above = partner_cum.get(row["ZONE_CODE"], {}).get(bb, {}).get("total", 0)
        con_partners = part_above * ppf

        s_w_edf = row["AVG_SILVER_FEES"] + (rev / row["SILVER_ORDERS"] if row["SILVER_ORDERS"] > 0 else 0)
        g_fees = row["AVG_GOLD_FEES_NO_EDF"] if row["AVG_GOLD_FEES_NO_EDF"] > 0 else row.get("AVG_GOLD_SF", 0)

        results.append({
            "Zone": row["ZONE_CODE"],
            "Silver Orders": int(row["SILVER_ORDERS"]),
            "Gold Orders": int(row.get("GOLD_ORDERS", 0)),
            "Partners": int(row["TOTAL_PARTNERS"]),
            "Avg Silver Fees": round(row["AVG_SILVER_FEES"], 2),
            "Avg Gold Fees": round(g_fees, 2),
            "Threshold (km)": round(b_km, 1),
            "Binds": constraint,
            "% Orders EDF": round(con_orders * 100.0 / row["SILVER_ORDERS"], 1) if row["SILVER_ORDERS"] > 0 else 0,
            "% Partners EDF": round(con_partners * 100.0 / row["TOTAL_PARTNERS"], 1) if row["TOTAL_PARTNERS"] > 0 else 0,
            "EDF Orders": int(round(con_orders)),
            "EDF Partners": int(round(con_partners)),
            "EDF Rev (3mo)": round(rev, 0),
            "Δ w/ EDF": round(s_w_edf - g_fees, 2),
            "Δ Current": round(row["AVG_SILVER_FEES"] - g_fees, 2),
            "_raw_edf": raw_edf,
            "_con_orders": con_orders,
            "_silver_fees_w_edf": s_w_edf,
            "_avg_gold_fees": g_fees,
            "_avg_silver_fees": row["AVG_SILVER_FEES"],
        })
    return pd.DataFrame(results)


# ── Load data ─────────────────────────────────────────────────────────────────

df, payg_df, partner_payg_df = load_data()

ORDER_PCT_COLS = ["PCT_ABOVE_099", "PCT_ABOVE_149", "PCT_ABOVE_199",
                  "PCT_ABOVE_249", "PCT_ABOVE_299", "PCT_ABOVE_399", "PCT_ABOVE_499"]
PARTNER_PCT_COLS = ["PCT_PART_ABOVE_099", "PCT_PART_ABOVE_149", "PCT_PART_ABOVE_199",
                    "PCT_PART_ABOVE_249", "PCT_PART_ABOVE_299", "PCT_PART_ABOVE_399",
                    "PCT_PART_ABOVE_499"]

payg_cum = build_payg_lookup(payg_df, "PAYG_ORDERS", ORDER_PCT_COLS)
partner_cum = build_payg_lookup(partner_payg_df, "PARTNERS", PARTNER_PCT_COLS)

# ── Sidebar controls ─────────────────────────────────────────────────────────

st.sidebar.markdown("## EDF Pricing")
edf_amount = st.sidebar.slider("Base Amount (£)", 0.49, 5.99, 1.99, 0.10, format="£%.2f")
num_tiers = st.sidebar.slider("Number of Tiers", 1, 3, 1)
tier_increment = st.sidebar.slider("Tier Increment (£)", 0.25, 2.00, 1.00, 0.25, format="£%.2f")

st.sidebar.markdown("## Targeting")
order_threshold = st.sidebar.slider("Order Threshold (%)", 5, 80, 40, 5,
                                    help="% of furthest orders targeted for EDF")
selection_cap = st.sidebar.slider("Selection Cap (%)", 20, 80, 50, 5,
                                  help="Max % of partners that can be impacted")

st.sidebar.markdown("---")
zones_to_show = st.sidebar.slider("Zones to show", 10, 500, 50, 10)

# ── Run scenario ─────────────────────────────────────────────────────────────

result = run_scenario(df, payg_cum, partner_cum, edf_amount, num_tiers,
                      tier_increment, order_threshold, selection_cap)

# ── Compute summary stats ────────────────────────────────────────────────────

total_silver = result["Silver Orders"].sum()
rev3 = result["EDF Rev (3mo)"].sum()
raw_edf = result["_raw_edf"].sum()
con_orders = result["_con_orders"].sum()

w_sf = (result["Silver Orders"] * result["_avg_silver_fees"]).sum() / total_silver
w_gf = (result["Silver Orders"] * result["_avg_gold_fees"]).sum() / total_silver
w_se = (result["Silver Orders"] * result["_silver_fees_w_edf"]).sum() / total_silver

current_delta = w_sf - w_gf
delta_w_edf = w_se - w_gf
uplift = delta_w_edf - current_delta
payg_reduction = (1 - con_orders / raw_edf) * 100 if raw_edf else 0

pct_orders_post = (result["Silver Orders"] * result["% Orders EDF"]).sum() / total_silver
pct_partners_post = (result["Silver Orders"] * result["% Partners EDF"]).sum() / total_silver
con_partners_total = result["EDF Partners"].sum()
sel_bound = (result["Binds"] == "Selection").sum()
ord_bound = len(result) - sel_bound

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("# Plus Silver EDF Scenario Model")

tier_desc = f"£{edf_amount:.2f}"
if num_tiers >= 2:
    tier_desc += f" / £{edf_amount + tier_increment:.2f}"
if num_tiers >= 3:
    tier_desc += f" / £{edf_amount + 2 * tier_increment:.2f}"

st.info(
    f"**EDF {tier_desc}** · {num_tiers} tier{'s' if num_tiers > 1 else ''} · "
    f"Furthest **{order_threshold}%** of orders · "
    f"Max **{selection_cap}%** selection impact"
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _short(v, prefix=""):
    """Format large numbers with shorthand: 1.8m, 245k, etc."""
    av = abs(v)
    if av >= 1_000_000:
        return f"{prefix}{v / 1_000_000:.1f}m"
    if av >= 100_000:
        return f"{prefix}{v / 1_000:.0f}k"
    if av >= 10_000:
        return f"{prefix}{v / 1_000:.1f}k"
    return f"{prefix}{v:,.0f}"

# ── Silver vs Gold Delta ─────────────────────────────────────────────────────

st.markdown("### Silver vs Gold Delta (order-weighted)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Current Delta", f"£{current_delta:.2f}")
c2.metric("Delta w/ EDF", f"£{delta_w_edf:.2f}", delta=f"£{uplift:.2f}")
c3.metric("Uplift", f"£{uplift:.2f}")
c4.metric("PAYG Constraint", f"{payg_reduction:.1f}%",
          help="% of distance-eligible orders removed by the PAYG DF >= EDF rule")

# ── Coverage ──────────────────────────────────────────────────────────────────

st.markdown("### Coverage")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("% Orders Paying EDF", f"{pct_orders_post:.1f}%")
c2.metric("% Partners w/ EDF", f"{pct_partners_post:.1f}%")
c3.metric("EDF Partners", _short(con_partners_total))
c4.metric("Selection-Bound", f"{sel_bound:,}")
c5.metric("Order-Bound", f"{ord_bound:,}")

# ── Zone distribution by order impact ────────────────────────────────────────

st.markdown("### Zone Distribution (% orders paying EDF)")

bins = list(range(0, 101, 10))
bin_labels = [f"{lo}-{hi}%" for lo, hi in zip(bins[:-1], bins[1:])]
result["_impact_bin"] = pd.cut(
    result["% Orders EDF"], bins=bins, labels=bin_labels,
    include_lowest=True, right=True,
)
dist = result["_impact_bin"].value_counts().reindex(bin_labels, fill_value=0)
dist_pct = (dist / len(result) * 100).round(1)

dist_cols = st.columns(len(bin_labels))
for col, label in zip(dist_cols, bin_labels):
    zone_count = int(dist[label])
    zone_pct = dist_pct[label]
    col.metric(label, f"{zone_count}", delta=f"{zone_pct:.0f}%", delta_color="off")

# ── Zone detail table ─────────────────────────────────────────────────────────

st.markdown("### Zone Detail")
st.caption(f"Top {zones_to_show} zones by Silver order volume")

display_cols = [
    "Zone", "Silver Orders", "Partners", "Avg Silver Fees", "Avg Gold Fees",
    "Threshold (km)", "Binds", "% Orders EDF", "% Partners EDF",
    "EDF Orders", "EDF Partners", "EDF Rev (3mo)", "Δ w/ EDF", "Δ Current",
]

display_df = result[display_cols].head(zones_to_show)

st.dataframe(
    display_df,
    use_container_width=True,
    height=600,
    column_config={
        "Zone": st.column_config.TextColumn("Zone", width="small"),
        "Silver Orders": st.column_config.NumberColumn("Silver Orders", format="%d"),
        "Partners": st.column_config.NumberColumn("Partners", format="%d"),
        "Avg Silver Fees": st.column_config.NumberColumn("Avg S Fees", format="£%.2f"),
        "Avg Gold Fees": st.column_config.NumberColumn("Avg G Fees", format="£%.2f"),
        "Threshold (km)": st.column_config.NumberColumn("Threshold", format="%.1f km"),
        "Binds": st.column_config.TextColumn("Binds", width="small"),
        "% Orders EDF": st.column_config.NumberColumn("% Ord EDF", format="%.1f%%"),
        "% Partners EDF": st.column_config.NumberColumn("% Part EDF", format="%.1f%%"),
        "EDF Orders": st.column_config.NumberColumn("EDF Ord", format="%d"),
        "EDF Partners": st.column_config.NumberColumn("EDF Part", format="%d"),
        "EDF Rev (3mo)": st.column_config.NumberColumn("EDF Rev", format="£%.0f"),
        "Δ w/ EDF": st.column_config.NumberColumn("Δ w/ EDF", format="£%.2f"),
        "Δ Current": st.column_config.NumberColumn("Δ Current", format="£%.2f"),
    },
    hide_index=True,
)

# ── Revenue Impact ────────────────────────────────────────────────────────────

st.markdown("### Revenue Impact")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Monthly", _short(rev3 / 3, "£"))
c2.metric("3-Month", _short(rev3, "£"))
c3.metric("Annualised", _short(rev3 * 4, "£"))
c4.metric("EDF Orders (3mo)", _short(con_orders))

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    f"Data: trailing 3 months · UKI (GB + IE) · {len(df):,} zones · "
    f"{_short(df['SILVER_ORDERS'].sum())} Silver orders · "
    f"{_short(df['GOLD_ORDERS'].sum())} Gold orders · "
    "Gold baseline: existing EDF removed"
)
