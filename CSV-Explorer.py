
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

st.set_page_config(page_title="CSV Explorer", layout="wide")
st.title("🔎 CSV Explorer — Upload · Filter · Visualize")

@st.cache_data
def load_csv(uploaded_file):
    # try common encodings
    for enc in ("utf-8", "utf-8-sig", "ISO-8859-1", "latin1"):
        try:
            return pd.read_csv(uploaded_file, encoding=enc)
        except Exception:
            uploaded_file.seek(0)
    # fallback: let pandas try default
    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file)

def summarize_df(df):
    st.subheader("📄 Quick summary")
    st.write(f"Rows: **{df.shape[0]}**, Columns: **{df.shape[1]}**")
    with st.expander("Column types & missing values"):
        types = pd.DataFrame({
            "dtype": df.dtypes.astype(str),
            "missing": df.isnull().sum(),
            "missing_pct": (df.isnull().mean() * 100).round(2),
            "unique_vals": df.nunique(dropna=False)
        })
        st.dataframe(types)

def build_filter_ui(df):
    """A simple filter builder: choose column, choose operator, choose value, add to active filters."""
    if "filters" not in st.session_state:
        st.session_state["filters"] = []

    with st.sidebar.form("filter_form", clear_on_submit=False):
        st.write("### 🛠️ Add filter")
        col = st.selectbox("Column", options=df.columns)
        dtype = df[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            op = st.selectbox("Operator", ["==", "!=", "<", "<=", ">", ">="])
            # show current min/max for convenience
            mn, mx = float(df[col].min(skipna=True)), float(df[col].max(skipna=True))
            val = st.number_input("Value", value=float(mn))
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            op = st.selectbox("Operator", ["==", "!=", "<", "<=", ">", ">="])
            val = st.date_input("Date")
        else:
            op = st.selectbox("Operator", ["==", "!=", "contains", "not contains", "in", "not in"])
            # for categorical, suggest top values
            top_vals = df[col].astype(str).value_counts().index.tolist()[:20]
            if op in ("in", "not in"):
                val = st.multiselect("Pick values", options=top_vals)
            else:
                val = st.text_input("Value")

        add = st.form_submit_button("Add filter")
        if add:
            st.session_state["filters"].append({"col": col, "op": op, "val": val})
    # display active filters and allow removal
    if st.session_state.get("filters"):
        st.write("**Active filters:**")
        for i, f in enumerate(st.session_state["filters"]):
            st.write(f"{i+1}. `{f['col']}` {f['op']} `{f['val']}`")
            if st.button(f"Remove filter {i+1}", key=f"rm{i}"):
                st.session_state["filters"].pop(i)
                st.experimental_rerun()
        if st.button("Clear all filters"):
            st.session_state["filters"] = []
            st.experimental_rerun()

def apply_filters(df):
    """Apply all filters stored in session_state['filters'] to the dataframe."""
    filters = st.session_state.get("filters", [])
    out = df.copy()
    for f in filters:
        col, op, val = f["col"], f["op"], f["val"]
        if pd.api.types.is_numeric_dtype(out[col].dtype):
            try:
                v = float(val)
            except Exception:
                # skip invalid numeric filter
                continue
            if op == "==":
                out = out[out[col] == v]
            elif op == "!=":
                out = out[out[col] != v]
            elif op == "<":
                out = out[out[col] < v]
            elif op == "<=":
                out = out[out[col] <= v]
            elif op == ">":
                out = out[out[col] > v]
            elif op == ">=":
                out = out[out[col] >= v]
        else:
            s = out[col].astype(str)
            if op == "==":
                out = out[s == str(val)]
            elif op == "!=":
                out = out[s != str(val)]
            elif op == "contains":
                out = out[s.str.contains(str(val), na=False, case=False)]
            elif op == "not contains":
                out = out[~s.str.contains(str(val), na=False, case=False)]
            elif op == "in":
                out = out[s.isin([str(x) for x in val])]
            elif op == "not in":
                out = out[~s.isin([str(x) for x in val])]
    return out

def viz_area(df):
    st.subheader("📊 Visualization")
    cols = df.columns.tolist()
    x = st.selectbox("X (categorical or numeric)", options=cols, index=0)
    y = st.selectbox("Y (numeric for aggregations / or leave blank)", options=[None] + cols, index=0)
    chart_type = st.selectbox("Chart type", options=["Bar (agg)", "Line (agg)", "Scatter", "Histogram"])

    # prepare data for plotting
    plot_df = df.copy()
    if chart_type in ("Bar (agg)", "Line (agg)"):
        # if y is numeric, aggregate
        if y and pd.api.types.is_numeric_dtype(plot_df[y]):
            agg = st.selectbox("Aggregation", options=["mean", "sum", "median", "count"], index=0)
            if agg == "mean":
                plot_df = plot_df.groupby(x, dropna=False)[y].mean().reset_index()
            elif agg == "sum":
                plot_df = plot_df.groupby(x, dropna=False)[y].sum().reset_index()
            elif agg == "median":
                plot_df = plot_df.groupby(x, dropna=False)[y].median().reset_index()
            elif agg == "count":
                plot_df = plot_df.groupby(x, dropna=False)[y].count().reset_index()
            fig = px.bar(plot_df, x=x, y=y, labels={x:x, y:y}, title=f"{agg} of {y} by {x}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            # y not given or not numeric: count per x
            plot_df = plot_df.groupby(x, dropna=False).size().reset_index(name="count")
            fig = px.bar(plot_df, x=x, y="count", labels={x:x, "count":"count"}, title=f"Count by {x}")
            st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Scatter":
        if y is None:
            st.info("For scatter please select an X and Y (both numeric recommended).")
        else:
            fig = px.scatter(plot_df, x=x, y=y, title=f"{y} vs {x}", labels={x:x, y:y})
            st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Histogram":
        if pd.api.types.is_numeric_dtype(plot_df[x]):
            fig = px.histogram(plot_df, x=x, nbins=30, title=f"Histogram of {x}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Histogram is best for numeric columns. Consider choosing a numeric X.")

def correlation_matrix(df):
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] >= 2:
        st.subheader("🔗 Correlation matrix")
        fig = px.imshow(num.corr(), text_auto=".2f", aspect="auto")
        st.plotly_chart(fig, use_container_width=True)

def download_button(df):
    st.subheader("⬇️ Download filtered data")
    towrite = BytesIO()
    df.to_csv(towrite, index=False, encoding="utf-8-sig")
    towrite.seek(0)
    st.download_button("Download CSV", data=towrite, file_name="filtered_data.csv", mime="text/csv")

def main():
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is None:
        st.info("Upload a CSV (or run this app and upload one) to begin.")
        return

    df = load_csv(uploaded_file)
    summarize_df(df)

    build_filter_ui(df)
    filtered = apply_filters(df)
    st.subheader("🔍 Filtered data (first 200 rows)")
    st.dataframe(filtered.head(200))

    viz_area(filtered)
    correlation_matrix(filtered)
    download_button(filtered)

if __name__ == "__main__":
    main()
