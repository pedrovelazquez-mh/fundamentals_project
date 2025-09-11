# one_factor="quality_score"
metodo_portafolio="binario" #"cuartil"
top_percentil=0.75   #en caso de elegir cuartil, se toma el percentil deseado
strategy_initial_time='2008-03-28'
corr_initial_time='2000-01-01'
ending_time='2025-04-01'
CAPITAL_INICIAL = 100
QUARTER_LAG      = 40
ANNUAL_LAG       = 60

min_empresas=2 #Minimo de empresas que componen el portafolio por sector 
alpha=0.1 #HRP
lag_by_sector={
    "Financials": [50,60],
    "Energy": [40,65] ,
    "Utilities": [40,65] ,
    "Basic Materials": [40,60] 
}
# lag_by_sector={
#     "Financials": [40,60],
#     "Energy": [40,60] ,
#     "Utilities": [40,60] ,
#     "Basic Materials": [40,60] 
# }
factor_weights = {
    "value_score": 0.50,
    "quality_score": 0.25,  
    "credit_score": 0.25
}
sector_weights={
    "Basic Materials": 0.1,
    "Financials": 0.3,
    "Energy": 0.45,
    "Utilities": 0.15
}


ticker_sector = {
    # ENERGY
    'YPFD': 'Energy',
    'PAMP': 'Energy',
    'CEPU': 'Energy',
    'CECO2': 'Energy',
    'CAPX': 'Energy',

      # UTILITIES
    'TGSU2': 'Utilities',
    'EDN':  'Utilities',
    'TRAN': 'Utilities',
    'TGNO4': 'Utilities',
    'CGPA2': 'Utilities',
    'METR': 'Utilities',
    'GBAN': 'Utilities',

    #BASIC MATERIALS
    'TXAR': 'Basic Materials',
    'CARC': 'Basic Materials',
    'ALUA': 'Basic Materials',
    'HARG': 'Basic Materials',
    'LOMA': 'Basic Materials',
    # BANKS 
    'SUPV': 'Financials',
    'BHIP': 'Financials',
    'BPAT': 'Financials',
    'BBAR': 'Financials',
    'GGAL': 'Financials',
    'BMA': 'Financials',
}

carpeta_fundamentals=r"C:\Users\Pedro\Research\Fundamentals\Bloomberg\bbg data\FA"
carpeta_prices=r"C:\Users\Pedro\Research\Fundamentals\Bloomberg\bbg data\prices"
carpeta_releasing=r"C:\Users\Pedro\Research\Fundamentals\Bloomberg\bbg data\ERN-MANUAL"
carpeta_hrp_metrics=r"C:\Users\Pedro\Research\Fundamentals\Bloomberg\Cross sectional\HRP_metrics"
W_min, W_max = 0.05, 0.15    #Winsorización  
factors = ["value_score", "quality_score", "credit_score"]
factor_meta = {
    # --- VALUE SCORE ---
    "AVERAGE_PRICE_EARNINGS_RATIO": {"factor": "value_score", "invert_sign": True},
    "AVERAGE_PRICE_TO_BOOK_RATIO": {"factor": "value_score", "invert_sign": True},
    "AVERAGE_PRICE_TO_CASH_FLOW": {"factor": "value_score", "invert_sign": True},
    "AVERAGE_PRICE_TO_FREE_CASH_FLOW": {"factor": "value_score", "invert_sign": True},
    "AVERAGE_PRICE_TO_SALES_RATIO": {"factor": "value_score", "invert_sign": True},
    "MKT_CAP_TO_DEPOSITS": {"factor": "value_score", "invert_sign": True},
    "AVG_EV_TO_T12M_EBITDA": {"factor": "value_score", "invert_sign": True},
    "AVERAGE_EV_TO_T12M_SALES": {"factor": "value_score", "invert_sign": True},
    "EV_TO_BOOK_VALUE": {"factor": "value_score", "invert_sign": True},
    # --- QUALITY SCORE ---
    "RETURN_COM_EQY": {"factor": "quality_score", "invert_sign": False},
    "RETURN_ON_ASSET": {"factor": "quality_score", "invert_sign": False},
    "RETURN_ON_INV_CAPITAL": {"factor": "quality_score", "invert_sign": False},
    "GROSS_MARGIN": {"factor": "quality_score", "invert_sign": False},
    "OPER_MARGIN": {"factor": "quality_score", "invert_sign": False},
    "PROF_MARGIN": {"factor": "quality_score", "invert_sign": False},
    # --- CREDIT SCORE ---
    "TOT_DEBT_TO_EBITDA": {"factor": "credit_score", "invert_sign": True},
    "EBITDA_TO_INTEREST_EXPN": {"factor": "credit_score", "invert_sign": False},
    "COM_EQY_TO_TOT_ASSET": {"factor": "credit_score", "invert_sign": False},
    "LT_DEBT_TO_TOT_ASSET": {"factor": "credit_score", "invert_sign": True},
    "TOT_DEBT_TO_TOT_ASSET": {"factor": "credit_score", "invert_sign": True},
    "CASH_FLOW_TO_TOT_LIAB": {"factor": "credit_score", "invert_sign": False},
    "CUR_RATIO": {"factor": "credit_score", "invert_sign": False},
    "QUICK_RATIO": {"factor": "credit_score", "invert_sign": False}
}

