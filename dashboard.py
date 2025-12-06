import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objects as go
import plotly.express as px
from recommend import recommend_for_text
import pandas as pd
from datetime import datetime
import json
import os
import torch

# Fix PyTorch meta tensor issue
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
torch.set_default_tensor_type(torch.FloatTensor)

# Set working directory to script location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Initialize the Dash app
app = dash.Dash(__name__, title="Scopus Recommender Dashboard")

# Store for search history
search_history = []

# Custom color scheme for dark mode
colors = {
    'background': '#0e1117',
    'card_bg': '#1e2130',
    'text': '#fafafa',
    'primary': '#4da6ff',
    'secondary': '#66b3ff',
    'high': '#4CAF50',    # Green for HIGH
    'medium': '#FFC107',  # Yellow for MEDIUM
    'low': '#FF5722',     # Red for LOW
    'border': '#2d3142'
}


# Custom CSS for placeholder styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            ::placeholder {
                color: rgba(250, 250, 250, 0.5) !important;
                opacity: 1;
            }
            :-ms-input-placeholder {
                color: rgba(250, 250, 250, 0.5) !important;
            }
            ::-ms-input-placeholder {
                color: rgba(250, 250, 250, 0.5) !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# App layout
app.layout = html.Div(style={'backgroundColor': colors['background'], 'minHeight': '100vh', 'padding': '20px'}, children=[
    # Header
    html.Div([
        html.H1('📚 Scopus Journal Recommender Dashboard', 
                style={'color': colors['primary'], 'textAlign': 'center', 'marginBottom': '10px'}),
        html.P('Find the perfect journal for your research paper with AI-powered recommendations',
               style={'color': colors['text'], 'textAlign': 'center', 'fontSize': '18px', 'marginBottom': '30px'})
    ]),
    
    # Main container
    html.Div([
        # Left panel - Input section
        html.Div([
            html.Div([
                html.H3('🔍 Search Parameters', style={'color': colors['secondary'], 'marginBottom': '20px'}),
                
                html.Label('📝 Paper Title', style={'color': colors['text'], 'fontWeight': 'bold', 'marginTop': '20px'}),
                dcc.Input(
                    id='title-input',
                    type='text',
                    placeholder='Enter your paper title...',
                    style={
                        'width': '100%',
                        'padding': '12px',
                        'marginTop': '12px',
                        'marginBottom': '10px',
                        'backgroundColor': colors['card_bg'],
                        'color': colors['text'],
                        'border': f'1px solid {colors["border"]}',
                        'borderRadius': '5px',
                        'fontSize': '14px'
                    },
                    className='custom-input'
                ),
                
                html.Label('📄 Abstract', style={'color': colors['text'], 'fontWeight': 'bold', 'marginTop': '30px'}),
                dcc.Textarea(
                    id='abstract-input',
                    placeholder='Enter your abstract...',
                    style={
                        'width': '100%',
                        'height': '150px',
                        'padding': '12px',
                        'marginTop': '12px',
                        'marginBottom': '10px',
                        'backgroundColor': colors['card_bg'],
                        'color': colors['text'],
                        'border': f'1px solid {colors["border"]}',
                        'borderRadius': '5px',
                        'fontSize': '14px',
                        'resize': 'vertical'
                    },
                    className='custom-textarea'
                ),
                
                html.Label('⚙️ Recommendation Level', style={'color': colors['text'], 'fontWeight': 'bold', 'marginTop': '30px'}),
                dcc.Dropdown(
                    id='level-dropdown',
                    options=[
                        {'label': 'Journal Level', 'value': 'journal'},
                        {'label': 'Paper Level', 'value': 'paper'}
                    ],
                    value='journal',
                    style={
                        'marginTop': '12px',
                        'marginBottom': '10px',
                        'backgroundColor': colors['card_bg'],
                        'color': colors['text'],
                        'borderRadius': '5px'
                    }
                ),
                
                html.Label('📊 Number of Results', style={'color': colors['text'], 'fontWeight': 'bold', 'marginTop': '30px'}),
                dcc.Slider(
                    id='top-k-slider',
                    min=5,
                    max=30,
                    step=5,
                    value=15,
                    marks={i: str(i) for i in range(5, 35, 5)},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
                
                html.Div(style={'marginTop': '20px'}),
                
                html.Button(
                    '🚀 Get Recommendations',
                    id='recommend-button',
                    n_clicks=0,
                    style={
                        'width': '100%',
                        'padding': '15px',
                        'marginTop': '35px',
                        'backgroundColor': colors['primary'],
                        'color': 'white',
                        'border': 'none',
                        'borderRadius': '5px',
                        'fontSize': '16px',
                        'fontWeight': 'bold',
                        'cursor': 'pointer'
                    }
                ),
                
                html.Div(id='status-message', style={'marginTop': '20px', 'textAlign': 'center'}),
                
                # Search History
                html.Hr(style={'borderColor': colors['border'], 'marginTop': '30px'}),
                html.H4('📜 Recent Searches', style={'color': colors['secondary'], 'marginTop': '20px'}),
                html.Div(id='search-history', style={'marginTop': '10px'})
                
            ], style={
                'backgroundColor': colors['card_bg'],
                'padding': '25px',
                'borderRadius': '10px',
                'border': f'1px solid {colors["border"]}',
                'boxShadow': '0 4px 6px rgba(0,0,0,0.3)'
            })
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '2%'}),
        
        # Right panel - Results section
        html.Div([
            # Summary metrics
            html.Div(id='summary-metrics', style={'marginBottom': '20px'}),
            
            # Tabs for different views
            dcc.Tabs(id='results-tabs', value='tab-overview', children=[
                dcc.Tab(label='📊 Overview', value='tab-overview', style={'backgroundColor': colors['card_bg'], 'color': colors['text']},
                        selected_style={'backgroundColor': colors['primary'], 'color': 'white'}),
                dcc.Tab(label='📈 Analytics', value='tab-analytics', style={'backgroundColor': colors['card_bg'], 'color': colors['text']},
                        selected_style={'backgroundColor': colors['primary'], 'color': 'white'}),
                dcc.Tab(label='📋 Data Table', value='tab-table', style={'backgroundColor': colors['card_bg'], 'color': colors['text']},
                        selected_style={'backgroundColor': colors['primary'], 'color': 'white'}),
                dcc.Tab(label='🎯 Details', value='tab-details', style={'backgroundColor': colors['card_bg'], 'color': colors['text']},
                        selected_style={'backgroundColor': colors['primary'], 'color': 'white'}),
            ], style={'marginBottom': '20px'}),
            
            html.Div(id='tab-content')
            
        ], style={'width': '68%', 'display': 'inline-block', 'verticalAlign': 'top'})
        
    ]),
    
    # Store for results
    dcc.Store(id='results-store'),
    dcc.Store(id='history-store', data=[]),
    
    # Hidden download components (always present)
    html.Div([
        html.Button(id='download-button', n_clicks=0, style={'display': 'none'}),
        html.Button(id='download-button-visible', n_clicks=0, style={'display': 'none'}),
        dcc.Download(id='download-csv')
    ], style={'display': 'none'})
])

# Callback for recommendations
@app.callback(
    [Output('results-store', 'data'),
     Output('status-message', 'children'),
     Output('history-store', 'data')],
    [Input('recommend-button', 'n_clicks')],
    [State('title-input', 'value'),
     State('abstract-input', 'value'),
     State('level-dropdown', 'value'),
     State('top-k-slider', 'value'),
     State('history-store', 'data')]
)
def get_recommendations(n_clicks, title, abstract, level, top_k, history):
    if n_clicks == 0:
        return None, '', history or []
    
    if not title or not title.strip():
        return None, html.Div('⚠️ Please provide a paper title.', 
                               style={'color': colors['low'], 'fontWeight': 'bold'}), history or []
    
    text = title.strip() + ". " + (abstract.strip() if abstract and abstract.strip().lower() != "not available" else "")
    paper_level = level == 'paper'
    use_classifier = level == 'journal'
    
    try:
        results = recommend_for_text(
            text,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            top_k=top_k,
            use_classifier=use_classifier,
            paper_level=paper_level
        )
        
        # Add to history
        history = history or []
        history.append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'title': title[:50],
            'level': 'Journal' if level == 'journal' else 'Paper',
            'count': len(results)
        })
        
        # Keep only last 5
        if len(history) > 5:
            history = history[-5:]
        
        return results, html.Div('✅ Recommendations generated successfully!', 
                                  style={'color': colors['high'], 'fontWeight': 'bold'}), history
    except Exception as e:
        return None, html.Div(f'❌ Error: {str(e)}', 
                               style={'color': colors['low'], 'fontWeight': 'bold'}), history or []

# Callback for search history display
@app.callback(
    Output('search-history', 'children'),
    [Input('history-store', 'data')]
)
def update_history(history):
    if not history:
        return html.P('No recent searches', style={'color': colors['text'], 'fontStyle': 'italic'})
    
    history_items = []
    for item in reversed(history):
        history_items.append(
            html.Div([
                html.P([
                    html.Strong(f"🕒 {item['timestamp']}", style={'color': colors['secondary']}),
                    html.Br(),
                    html.Span(f"Title: {item['title']}...", style={'fontSize': '12px'}),
                    html.Br(),
                    html.Span(f"Level: {item['level']} | Results: {item['count']}", 
                             style={'fontSize': '11px', 'color': '#999'})
                ], style={'marginBottom': '10px', 'padding': '10px', 'backgroundColor': colors['background'],
                         'borderRadius': '5px', 'border': f'1px solid {colors["border"]}'})
            ])
        )
    
    return html.Div(history_items)

# Callback for summary metrics
@app.callback(
    Output('summary-metrics', 'children'),
    [Input('results-store', 'data')]
)
def update_metrics(results):
    if not results:
        return html.Div([
            html.H3('👋 Welcome!', style={'color': colors['text'], 'textAlign': 'center'}),
            html.P('Enter your paper details and click "Get Recommendations" to start',
                   style={'color': colors['text'], 'textAlign': 'center', 'marginTop': '20px'})
        ], style={
            'backgroundColor': colors['card_bg'],
            'padding': '40px',
            'borderRadius': '10px',
            'border': f'1px solid {colors["border"]}'
        })
    
    df = pd.DataFrame(results)
    high_count = len(df[df['confidence'] == 'HIGH'])
    
    return html.Div([
        html.Div([
            html.Div([
                html.H4('🏆 Top Journal', style={'color': colors['text'], 'marginBottom': '5px', 'fontSize': '14px'}),
                html.H3(df.iloc[0]['journal'], style={'color': colors['primary'], 'margin': '0'})
            ], style={
                'backgroundColor': colors['card_bg'],
                'padding': '20px',
                'borderRadius': '8px',
                'textAlign': 'center',
                'border': f'2px solid {colors["primary"]}',
                'width': '23%',
                'display': 'inline-block',
                'marginRight': '2%'
            }),
            
            html.Div([
                html.H4('⭐ Highest Score', style={'color': colors['text'], 'marginBottom': '5px', 'fontSize': '14px'}),
                html.H3(f"{df['score'].max():.3f}", style={'color': colors['high'], 'margin': '0'})
            ], style={
                'backgroundColor': colors['card_bg'],
                'padding': '20px',
                'borderRadius': '8px',
                'textAlign': 'center',
                'border': f'2px solid {colors["high"]}',
                'width': '23%',
                'display': 'inline-block',
                'marginRight': '2%'
            }),
            
            html.Div([
                html.H4('📈 Total Results', style={'color': colors['text'], 'marginBottom': '5px', 'fontSize': '14px'}),
                html.H3(str(len(df)), style={'color': colors['secondary'], 'margin': '0'})
            ], style={
                'backgroundColor': colors['card_bg'],
                'padding': '20px',
                'borderRadius': '8px',
                'textAlign': 'center',
                'border': f'2px solid {colors["secondary"]}',
                'width': '23%',
                'display': 'inline-block',
                'marginRight': '2%'
            }),
            
            html.Div([
                html.H4('🎯 High Confidence', style={'color': colors['text'], 'marginBottom': '5px', 'fontSize': '14px'}),
                html.H3(str(high_count), style={'color': colors['high'], 'margin': '0'})
            ], style={
                'backgroundColor': colors['card_bg'],
                'padding': '20px',
                'borderRadius': '8px',
                'textAlign': 'center',
                'border': f'2px solid {colors["high"]}',
                'width': '23%',
                'display': 'inline-block'
            })
        ])
    ])

# Callback for tab content
@app.callback(
    Output('tab-content', 'children'),
    [Input('results-tabs', 'value'),
     Input('results-store', 'data')]
)
def render_tab_content(tab, results):
    if not results:
        return html.Div()
    
    df = pd.DataFrame(results)
    
    # Color mapping for consistent colors: HIGH=Green, MEDIUM=Yellow, LOW=Red
    color_map = {'HIGH': colors['high'], 'MEDIUM': colors['medium'], 'LOW': colors['low']}
    
    if tab == 'tab-overview':
        # Score distribution scatter plot
        fig_scatter = go.Figure()
        
        for conf in ['HIGH', 'MEDIUM', 'LOW']:
            df_conf = df[df['confidence'] == conf]
            if not df_conf.empty:
                fig_scatter.add_trace(go.Scatter(
                    x=list(range(len(df_conf))),
                    y=df_conf['score'],
                    mode='markers',
                    name=conf,
                    marker=dict(
                        size=12,
                        color=colors['high'] if conf == 'HIGH' else colors['medium'] if conf == 'MEDIUM' else colors['low'],
                        opacity=0.8
                    ),
                    hovertemplate='<b>%{text}</b><br>Score: %{y:.4f}<extra></extra>',
                    text=df_conf['paper_title']
                ))
        
        fig_scatter.update_layout(
            title='Score Distribution Across Recommendations',
            xaxis_title='Rank',
            yaxis_title='Similarity Score',
            plot_bgcolor=colors['card_bg'],
            paper_bgcolor=colors['card_bg'],
            font=dict(color=colors['text']),
            hovermode='closest',
            showlegend=True,
            height=400
        )
        
        # Confidence pie chart
        conf_counts = df['confidence'].value_counts()
        # Ensure consistent color ordering: HIGH=Green, MEDIUM=Yellow, LOW=Red
        color_map = {'HIGH': colors['high'], 'MEDIUM': colors['medium'], 'LOW': colors['low']}
        pie_colors = [color_map.get(label, colors['medium']) for label in conf_counts.index]
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=conf_counts.index,
            values=conf_counts.values,
            marker=dict(colors=pie_colors),
            hole=0.4,
            textfont=dict(size=14, color='white')
        )])
        
        fig_pie.update_layout(
            title='Confidence Level Distribution',
            plot_bgcolor=colors['card_bg'],
            paper_bgcolor=colors['card_bg'],
            font=dict(color=colors['text']),
            height=400
        )
        
        return html.Div([
            html.Div([
                dcc.Graph(figure=fig_scatter)
            ], style={'width': '100%', 'display': 'inline-block', 'backgroundColor': colors['card_bg'],
                     'padding': '15px', 'borderRadius': '10px', 'marginBottom': '20px'}),
            
            html.Div([
                html.Div([
                    dcc.Graph(figure=fig_pie)
                ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
                
                html.Div([
                    dcc.Graph(figure=create_top_journals_bar(df))
                ], style={'width': '50%', 'display': 'inline-block'})
            ], style={'backgroundColor': colors['card_bg'], 'padding': '15px', 'borderRadius': '10px'})
        ])
    
    elif tab == 'tab-analytics':
        # Score histogram
        fig_hist = go.Figure(data=[go.Histogram(
            x=df['score'],
            nbinsx=20,
            marker=dict(color=colors['primary'], opacity=0.7),
            hovertemplate='Score Range: %{x}<br>Count: %{y}<extra></extra>'
        )])
        
        fig_hist.update_layout(
            title='Score Frequency Distribution',
            xaxis_title='Score Range',
            yaxis_title='Frequency',
            plot_bgcolor=colors['card_bg'],
            paper_bgcolor=colors['card_bg'],
            font=dict(color=colors['text']),
            height=350
        )
        
        # Average score by confidence
        avg_scores = df.groupby('confidence')['score'].mean().reset_index()
        # Map colors correctly: HIGH=Green, MEDIUM=Yellow, LOW=Red
        bar_colors = [color_map.get(conf, colors['medium']) for conf in avg_scores['confidence']]
        
        fig_bar = go.Figure(data=[go.Bar(
            x=avg_scores['confidence'],
            y=avg_scores['score'],
            marker=dict(color=bar_colors),
            text=avg_scores['score'].apply(lambda x: f'{x:.4f}'),
            textposition='auto',
        )])
        
        fig_bar.update_layout(
            title='Average Score by Confidence Level',
            xaxis_title='Confidence Level',
            yaxis_title='Average Score',
            plot_bgcolor=colors['card_bg'],
            paper_bgcolor=colors['card_bg'],
            font=dict(color=colors['text']),
            height=350
        )
        
        # Publisher distribution
        fig_pub = create_publisher_bar(df)
        
        return html.Div([
            html.Div([
                dcc.Graph(figure=fig_hist)
            ], style={'width': '100%', 'backgroundColor': colors['card_bg'], 'padding': '15px',
                     'borderRadius': '10px', 'marginBottom': '20px'}),
            
            html.Div([
                html.Div([
                    dcc.Graph(figure=fig_bar)
                ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
                
                html.Div([
                    dcc.Graph(figure=fig_pub)
                ], style={'width': '50%', 'display': 'inline-block'})
            ], style={'backgroundColor': colors['card_bg'], 'padding': '15px', 'borderRadius': '10px'})
        ])
    
    elif tab == 'tab-table':
        return html.Div([
            html.Div([
                html.H3('📋 Detailed Results', style={'color': colors['secondary'], 'marginBottom': '15px'}),
                
                dash_table.DataTable(
                    data=df.to_dict('records'),
                    columns=[
                        {'name': 'Paper Title', 'id': 'paper_title'},
                        {'name': 'Journal', 'id': 'journal'},
                        {'name': 'Publisher', 'id': 'publisher'},
                        {'name': 'Score', 'id': 'score', 'type': 'numeric', 'format': {'specifier': '.4f'}},
                        {'name': 'Confidence', 'id': 'confidence'},
                        {'name': 'Scopus ID', 'id': 'scopus_id'}
                    ],
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'backgroundColor': colors['background'],
                        'color': colors['text'],
                        'border': f'1px solid {colors["border"]}',
                        'textAlign': 'left',
                        'padding': '12px',
                        'fontSize': '13px'
                    },
                    style_header={
                        'backgroundColor': colors['card_bg'],
                        'fontWeight': 'bold',
                        'border': f'1px solid {colors["border"]}',
                        'fontSize': '14px'
                    },
                    style_data_conditional=[
                        {
                            'if': {'filter_query': '{confidence} = "HIGH"'},
                            'backgroundColor': f'{colors["high"]}22',
                        },
                        {
                            'if': {'filter_query': '{confidence} = "MEDIUM"'},
                            'backgroundColor': f'{colors["medium"]}22',
                        },
                        {
                            'if': {'filter_query': '{confidence} = "LOW"'},
                            'backgroundColor': f'{colors["low"]}22',
                        }
                    ],
                    page_size=15,
                    sort_action='native',
                    filter_action='native'
                ),
                
                html.Div([
                    html.Button(
                        '⬇️ Download CSV',
                        id='download-button-trigger',
                        n_clicks=0,
                        style={
                            'marginTop': '20px',
                            'padding': '12px 24px',
                            'backgroundColor': colors['primary'],
                            'color': 'white',
                            'border': 'none',
                            'borderRadius': '5px',
                            'cursor': 'pointer',
                            'fontSize': '14px',
                            'fontWeight': 'bold'
                        }
                    )
                ], id='download-button-visible')
                
            ], style={'backgroundColor': colors['card_bg'], 'padding': '20px', 'borderRadius': '10px'})
        ])
    
    elif tab == 'tab-details':
        details = []
        
        for conf in ['HIGH', 'MEDIUM', 'LOW']:
            df_conf = df[df['confidence'] == conf]
            if df_conf.empty:
                continue
            
            conf_color = colors['high'] if conf == 'HIGH' else colors['medium'] if conf == 'MEDIUM' else colors['low']
            conf_icon = '🟢' if conf == 'HIGH' else '🟡' if conf == 'MEDIUM' else '🔴'
            
            details.append(html.Div([
                html.H3(f'{conf_icon} {conf} Confidence ({len(df_conf)} results)', 
                       style={'color': conf_color, 'marginBottom': '15px'}),
                
                html.Div([
                    html.Div([
                        html.H4(f"📄 {row['paper_title']}", 
                               style={'color': colors['text'], 'marginBottom': '10px'}),
                        html.P([
                            html.Strong('📰 Journal: ', style={'color': colors['secondary']}),
                            html.Span(row['journal'], style={'color': colors['text']})
                        ], style={'marginBottom': '5px'}),
                        html.P([
                            html.Strong('🏢 Publisher: ', style={'color': colors['secondary']}),
                            html.Span(row['publisher'], style={'color': colors['text']})
                        ], style={'marginBottom': '5px'}),
                        html.P([
                            html.Strong('🆔 Scopus ID: ', style={'color': colors['secondary']}),
                            html.Span(row['scopus_id'], style={'color': colors['text']})
                        ], style={'marginBottom': '5px'}),
                        html.P([
                            html.Strong('⭐ Score: ', style={'color': colors['secondary']}),
                            html.Span(f"{row['score']:.4f}", style={'color': colors['text']})
                        ], style={'marginBottom': '5px'}),
                    ], style={
                        'backgroundColor': colors['card_bg'],
                        'padding': '20px',
                        'borderRadius': '8px',
                        'marginBottom': '15px',
                        'borderLeft': f'4px solid {conf_color}',
                        'boxShadow': '0 2px 4px rgba(0,0,0,0.2)'
                    }) for idx, row in df_conf.iterrows()
                ])
            ], style={'marginBottom': '30px'}))
        
        return html.Div(details, style={'backgroundColor': colors['card_bg'], 'padding': '20px', 'borderRadius': '10px'})

# Helper function for top journals bar chart
def create_top_journals_bar(df):
    top_journals = df['journal'].value_counts().head(10).reset_index()
    top_journals.columns = ['journal', 'count']
    
    fig = go.Figure(data=[go.Bar(
        x=top_journals['count'],
        y=top_journals['journal'],
        orientation='h',
        marker=dict(color=colors['secondary'], opacity=0.8),
        text=top_journals['count'],
        textposition='auto'
    )])
    
    fig.update_layout(
        title='Top 10 Journals',
        xaxis_title='Number of Recommendations',
        yaxis_title='Journal',
        plot_bgcolor=colors['card_bg'],
        paper_bgcolor=colors['card_bg'],
        font=dict(color=colors['text']),
        height=400,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

# Helper function for publisher bar chart
def create_publisher_bar(df):
    top_pubs = df['publisher'].value_counts().head(8).reset_index()
    top_pubs.columns = ['publisher', 'count']
    
    fig = go.Figure(data=[go.Bar(
        x=top_pubs['count'],
        y=top_pubs['publisher'],
        orientation='h',
        marker=dict(color=colors['medium'], opacity=0.8),
        text=top_pubs['count'],
        textposition='auto'
    )])
    
    fig.update_layout(
        title='Top 8 Publishers',
        xaxis_title='Number of Recommendations',
        yaxis_title='Publisher',
        plot_bgcolor=colors['card_bg'],
        paper_bgcolor=colors['card_bg'],
        font=dict(color=colors['text']),
        height=350,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

# Callback for CSV download
@app.callback(
    Output('download-csv', 'data'),
    [Input('download-button', 'n_clicks'),
     Input('download-button-trigger', 'n_clicks')],
    [State('results-store', 'data')],
    prevent_initial_call=True
)
def download_csv(n_clicks_hidden, n_clicks_trigger, results):
    if results:
        df = pd.DataFrame(results)
        return dcc.send_data_frame(df.to_csv, 
                                   f"scopus_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                   index=False)

if __name__ == '__main__':
    print("🚀 Starting Scopus Recommender Dashboard...")
    print("📊 Open your browser at: http://127.0.0.1:8050")
    app.run(debug=True, host='127.0.0.1', port=8050)
