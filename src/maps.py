"""
Geographic visualizations of Beige Book sector sentiment.

Maps Federal Reserve district sentiment onto US state choropleth maps
using Plotly for interactivity.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import DISTRICTS, DISTRICT_STATES, STATE_TO_DISTRICT, OUTPUT_DIR


# Full state names for hover labels
STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana",
    "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan",
    "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
    "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
    "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}


def _build_state_sector_data(sector_df, sector, date=None):
    """
    Build a state-level DataFrame for a given sector.

    Parameters
    ----------
    sector_df : pandas.core.frame.DataFrame
        Sector-level data with columns: date, district, sector, vader_compound.
    sector : str
        Sector to filter on.
    date : str or None
        If provided, filter to a specific date. Otherwise, average over all dates.

    Returns
    -------
    state_data : pandas.core.frame.DataFrame
        Columns: state, state_name, district, vader_compound.
    """
    df = sector_df[sector_df["sector"] == sector].copy()
    if date is not None:
        df = df[df["date"] == pd.to_datetime(date)]

    # Average sentiment by district
    district_avg = df.groupby("district")["vader_compound"].mean().to_dict()

    rows = []
    for state, district in STATE_TO_DISTRICT.items():
        sentiment = district_avg.get(district)
        if sentiment is not None:
            rows.append({
                "state": state,
                "state_name": STATE_NAMES.get(state, state),
                "district": district,
                "vader_compound": sentiment,
            })

    return pd.DataFrame(rows)


def plot_sector_map(sector_df, sector, date=None, save=True):
    """
    Choropleth map of the US colored by sector sentiment per Fed district.

    Parameters
    ----------
    sector_df : pandas.core.frame.DataFrame
        Sector-level data with columns: date, district, sector, vader_compound.
    sector : str
        Which sector to map.
    date : str or None
        Specific Beige Book date to show. If None, averages across all dates.
    save : bool
        Save as interactive HTML file.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    state_data = _build_state_sector_data(sector_df, sector, date)
    if state_data.empty:
        print(f"No data for sector '{sector}'")
        return None

    date_label = f" ({date})" if date else " (2011–2025 Average)"

    fig = go.Figure(go.Choropleth(
        locations=state_data["state"],
        z=state_data["vader_compound"],
        locationmode="USA-states",
        colorscale="RdYlGn",
        zmid=0,
        zmin=-0.5,
        zmax=0.8,
        colorbar_title="Sentiment",
        text=state_data.apply(
            lambda r: (
                f"{r['state_name']}<br>"
                f"District: {r['district']}<br>"
                f"Sentiment: {r['vader_compound']:.3f}"
            ), axis=1
        ),
        hoverinfo="text",
    ))

    fig.update_layout(
        title_text=f"Beige Book {sector} Sentiment by Fed District{date_label}",
        geo_scope="usa",
        geo=dict(
            showlakes=True,
            lakecolor="rgb(200, 220, 240)",
        ),
        width=1000,
        height=600,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = sector.lower().replace(" ", "_").replace("&", "and")
        path = OUTPUT_DIR / f"map_{safe_name}.html"
        fig.write_html(str(path))
        print(f"Saved: {path}")

    return fig


def plot_sector_map_grid(sector_df, sectors=None, save=True):
    """
    Grid of choropleth maps showing multiple sectors side by side.

    Parameters
    ----------
    sector_df : pandas.core.frame.DataFrame
        Sector-level data with columns: date, district, sector, vader_compound.
    sectors : list of str or None
        Which sectors to include. If None, uses top 6 by observation count.
    save : bool

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    if sectors is None:
        top = sector_df["sector"].value_counts().head(6).index.tolist()
        sectors = top

    cols = 2
    rows = (len(sectors) + cols - 1) // cols

    fig = make_subplots(
        rows=rows,
        cols=cols,
        specs=[[{"type": "choropleth"} for _ in range(cols)] for _ in range(rows)],
        subplot_titles=sectors,
    )

    for idx, sector in enumerate(sectors):
        row = idx // cols + 1
        col = idx % cols + 1

        state_data = _build_state_sector_data(sector_df, sector)
        if state_data.empty:
            continue

        fig.add_trace(
            go.Choropleth(
                locations=state_data["state"],
                z=state_data["vader_compound"],
                locationmode="USA-states",
                colorscale="RdYlGn",
                zmid=0,
                zmin=-0.5,
                zmax=0.8,
                colorbar=dict(title="Sentiment", len=0.3, y=1 - (row - 0.5) / rows),
                text=state_data.apply(
                    lambda r: (
                        f"{r['state_name']}<br>"
                        f"District: {r['district']}<br>"
                        f"{sector}: {r['vader_compound']:.3f}"
                    ), axis=1
                ),
                hoverinfo="text",
                showscale=(col == cols),
            ),
            row=row,
            col=col,
        )

    # Update each geo subplot to show USA
    for i in range(len(sectors)):
        geo_key = f"geo{i + 1}" if i > 0 else "geo"
        fig.update_layout(**{
            geo_key: dict(scope="usa", showlakes=True,
                          lakecolor="rgb(200, 220, 240)"),
        })

    fig.update_layout(
        title_text="Beige Book Sector Sentiment Across Federal Reserve Districts (2011–2025)",
        height=400 * rows,
        width=1100,
        margin=dict(l=10, r=10, t=80, b=10),
    )

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / "map_sector_grid.html"
        fig.write_html(str(path))
        print(f"Saved: {path}")

    return fig


def plot_dominant_sector_map(sector_df, mode="strongest", save=True):
    """
    Map showing each district's dominant sector (most positive or negative).

    Parameters
    ----------
    sector_df : pandas.core.frame.DataFrame
        Sector-level data with columns: date, district, sector, vader_compound.
    mode : str
        'strongest' for highest sentiment sector, 'weakest' for lowest.
    save : bool

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    # Exclude "General" — not a real sector
    df = sector_df[sector_df["sector"] != "General"]

    # Find dominant sector per district
    district_sector = (
        df.groupby(["district", "sector"])["vader_compound"]
        .mean()
        .reset_index()
    )

    if mode == "strongest":
        dominant = district_sector.loc[
            district_sector.groupby("district")["vader_compound"].idxmax()
        ]
    else:
        dominant = district_sector.loc[
            district_sector.groupby("district")["vader_compound"].idxmin()
        ]

    dominant_dict = dominant.set_index("district")[["sector", "vader_compound"]].to_dict("index")

    # Get unique sectors for color mapping
    unique_sectors = sorted(dominant["sector"].unique())
    sector_to_num = {s: i for i, s in enumerate(unique_sectors)}

    rows = []
    for state, district in STATE_TO_DISTRICT.items():
        info = dominant_dict.get(district)
        if info:
            rows.append({
                "state": state,
                "state_name": STATE_NAMES.get(state, state),
                "district": district,
                "sector": info["sector"],
                "vader_compound": info["vader_compound"],
                "sector_num": sector_to_num[info["sector"]],
            })

    state_data = pd.DataFrame(rows)

    mode_label = "Strongest" if mode == "strongest" else "Weakest"

    fig = go.Figure(go.Choropleth(
        locations=state_data["state"],
        z=state_data["sector_num"],
        locationmode="USA-states",
        colorscale="Viridis",
        colorbar=dict(
            title="Sector",
            tickvals=list(range(len(unique_sectors))),
            ticktext=unique_sectors,
        ),
        text=state_data.apply(
            lambda r: (
                f"{r['state_name']}<br>"
                f"District: {r['district']}<br>"
                f"{mode_label} Sector: {r['sector']}<br>"
                f"Sentiment: {r['vader_compound']:.3f}"
            ), axis=1
        ),
        hoverinfo="text",
    ))

    fig.update_layout(
        title_text=f"{mode_label} Sector by Federal Reserve District (2011–2025 Average)",
        geo_scope="usa",
        geo=dict(showlakes=True, lakecolor="rgb(200, 220, 240)"),
        width=1000,
        height=600,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / f"map_dominant_{mode}.html"
        fig.write_html(str(path))
        print(f"Saved: {path}")

    return fig


def plot_sector_map_animated(sector_df, sector, save=True):
    """
    Animated choropleth map showing sector sentiment evolving over time.

    Parameters
    ----------
    sector_df : pandas.core.frame.DataFrame
        Sector-level data with columns: date, district, sector, vader_compound.
    sector : str
        Which sector to animate.
    save : bool

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    df = sector_df[sector_df["sector"] == sector].copy()
    df["date"] = pd.to_datetime(df["date"])
    dates = sorted(df["date"].unique())

    frames = []
    for date in dates:
        state_data = _build_state_sector_data(sector_df, sector, date=date)
        if state_data.empty:
            continue

        frames.append(go.Frame(
            data=[go.Choropleth(
                locations=state_data["state"],
                z=state_data["vader_compound"],
                locationmode="USA-states",
                colorscale="RdYlGn",
                zmid=0,
                zmin=-0.5,
                zmax=0.8,
                text=state_data.apply(
                    lambda r: (
                        f"{r['state_name']}<br>"
                        f"District: {r['district']}<br>"
                        f"Sentiment: {r['vader_compound']:.3f}"
                    ), axis=1
                ),
                hoverinfo="text",
            )],
            name=str(date.date()),
        ))

    # Initial frame
    init_data = _build_state_sector_data(sector_df, sector, date=dates[0])

    fig = go.Figure(
        data=[go.Choropleth(
            locations=init_data["state"],
            z=init_data["vader_compound"],
            locationmode="USA-states",
            colorscale="RdYlGn",
            zmid=0,
            zmin=-0.5,
            zmax=0.8,
            colorbar_title="Sentiment",
            text=init_data.apply(
                lambda r: (
                    f"{r['state_name']}<br>"
                    f"District: {r['district']}<br>"
                    f"Sentiment: {r['vader_compound']:.3f}"
                ), axis=1
            ),
            hoverinfo="text",
        )],
        frames=frames,
    )

    # Add slider and play button
    fig.update_layout(
        title_text=f"Beige Book {sector} Sentiment Over Time",
        geo_scope="usa",
        geo=dict(showlakes=True, lakecolor="rgb(200, 220, 240)"),
        width=1000,
        height=650,
        margin=dict(l=20, r=20, t=60, b=20),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=0,
            x=0.5,
            xanchor="center",
            buttons=[
                dict(label="Play",
                     method="animate",
                     args=[None, {"frame": {"duration": 500, "redraw": True},
                                  "fromcurrent": True}]),
                dict(label="Pause",
                     method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate"}]),
            ],
        )],
        sliders=[dict(
            active=0,
            steps=[dict(args=[[f.name], {"frame": {"duration": 300, "redraw": True},
                                         "mode": "immediate"}],
                         label=f.name,
                         method="animate")
                   for f in frames],
            x=0.1,
            len=0.8,
            y=-0.05,
            currentvalue=dict(prefix="Date: ", visible=True),
        )],
    )

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = sector.lower().replace(" ", "_").replace("&", "and")
        path = OUTPUT_DIR / f"map_{safe_name}_animated.html"
        fig.write_html(str(path))
        print(f"Saved: {path}")

    return fig
