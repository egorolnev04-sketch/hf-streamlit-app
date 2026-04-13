import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from datetime import datetime

st.set_page_config(page_title="CHF Prediction System", layout="wide")

@st.cache_data
def load_data():
    chf_data = fetch_openml("Predicting-Critical-Heat-Flux", version=1, as_frame=True, parser='auto')
    df = chf_data.frame
    return df

@st.cache_data
def prepare_features(df):
    df = df.dropna()
    categorical_cols = ['author', 'geometry']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col])
    df['pressure_massflux'] = df['pressure_[MPa]'] * df['mass_flux_[kg/m2-s]']
    numeric_features = ['pressure_[MPa]', 'mass_flux_[kg/m2-s]', 'x_e_out_[-]', 
                        'D_e_[mm]', 'D_h_[mm]', 'length_[mm]', 'pressure_massflux']
    categorical_features = ['author_encoded', 'geometry_encoded']
    feature_names = numeric_features + categorical_features
    X = df[feature_names]
    y = df['chf_exp_[MW/m2]']
    return X, y, feature_names

@st.cache_resource
def train_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    metrics = {
        'R2': r2_score(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'MAPE': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    }
    return model, scaler, metrics, y_test, y_pred

@st.cache_resource
def train_gradient_boosting(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    metrics = {
        'R2': r2_score(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'MAPE': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    }
    return model, scaler, metrics, y_test, y_pred

@st.cache_resource
def train_linear_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    metrics = {
        'R2': r2_score(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'MAPE': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    }
    return model, scaler, metrics, y_test, y_pred

def get_safety_level(chf):
    if chf > 2.0:
        return "CRITICAL", "Немедленное внимание"
    elif chf > 1.0:
        return "NORMAL", "Нормальная работа"
    else:
        return "LOW", "Безопасный режим"

def main():
    st.title("Прогнозирование критического теплового потока (CHF)")
    st.markdown("---")

    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "Random Forest"
    if 'sidebar_filters' not in st.session_state:
        st.session_state.sidebar_filters = {'pressure_range': (0.0, 5.0), 'mass_flux_range': (0.0, 10000.0)}

    with st.spinner("Загрузка данных..."):
        df = load_data()
        X, y, feature_names = prepare_features(df)

    tab1, tab2, tab3, tab4 = st.tabs(["Данные", "Физический анализ", "Модели", "Прогноз CHF"])

    with tab1:
        st.header("Информация о датасете")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Количество записей", df.shape[0])
        with col2:
            st.metric("Количество признаков", df.shape[1])
        with col3:
            st.metric("Диапазон CHF", f"{df['chf_exp_[MW/m2]'].min():.2f} - {df['chf_exp_[MW/m2]'].max():.2f} MW/m²")
        st.subheader("Первые 5 строк")
        st.dataframe(df.head())
        st.subheader("Статистика")
        st.dataframe(df.describe())

    with tab2:
        st.header("Физический анализ зависимостей")
        col1, col2 = st.columns([1, 3])
        with col1:
            st.subheader("Фильтры")
            pressure_range = st.slider("Давление (МПа)", 0.0, 5.0, 
                                       st.session_state.sidebar_filters['pressure_range'], step=0.1)
            mass_flux_range = st.slider("Массовый расход (кг/м²·с)", 0.0, 10000.0,
                                        st.session_state.sidebar_filters['mass_flux_range'], step=100.0)
            st.session_state.sidebar_filters['pressure_range'] = pressure_range
            st.session_state.sidebar_filters['mass_flux_range'] = mass_flux_range
        filtered_df = df[(df['pressure_[MPa]'] >= pressure_range[0]) & 
                         (df['pressure_[MPa]'] <= pressure_range[1]) &
                         (df['mass_flux_[kg/m2-s]'] >= mass_flux_range[0]) & 
                         (df['mass_flux_[kg/m2-s]'] <= mass_flux_range[1])]
        with col2:
            plot_type = st.selectbox("Тип графика", ["3D Scatter", "Heatmap корреляций", "Box plot по давлению"])
            if plot_type == "3D Scatter":
                fig = px.scatter_3d(filtered_df, x='pressure_[MPa]', y='mass_flux_[kg/m2-s]', 
                                    z='chf_exp_[MW/m2]', color='chf_exp_[MW/m2]',
                                    title="CHF vs Pressure vs Mass Flux")
                st.plotly_chart(fig, use_container_width=True)
            elif plot_type == "Heatmap корреляций":
                corr = filtered_df[['pressure_[MPa]', 'mass_flux_[kg/m2-s]', 'x_e_out_[-]', 
                                    'D_e_[mm]', 'D_h_[mm]', 'length_[mm]', 'chf_exp_[MW/m2]']].corr()
                fig = px.imshow(corr, text_auto=True, title="Корреляционная матрица")
                st.plotly_chart(fig, use_container_width=True)
            else:
                filtered_df['pressure_bin'] = pd.cut(filtered_df['pressure_[MPa]'], bins=5)
                fig = px.box(filtered_df, x='pressure_bin', y='chf_exp_[MW/m2]', 
                             title="Распределение CHF по диапазонам давления")
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("Сравнение ML-моделей")
        with st.spinner("Обучение моделей..."):
            rf_model, rf_scaler, rf_metrics, _, _ = train_random_forest(X, y)
            gb_model, gb_scaler, gb_metrics, _, _ = train_gradient_boosting(X, y)
            lr_model, lr_scaler, lr_metrics, _, _ = train_linear_regression(X, y)
        metrics_df = pd.DataFrame({
            "Модель": ["Random Forest", "Gradient Boosting", "Linear Regression"],
            "R²": [rf_metrics['R2'], gb_metrics['R2'], lr_metrics['R2']],
            "RMSE (MW/m²)": [rf_metrics['RMSE'], gb_metrics['RMSE'], lr_metrics['RMSE']],
            "MAE (MW/m²)": [rf_metrics['MAE'], gb_metrics['MAE'], lr_metrics['MAE']],
            "MAPE (%)": [rf_metrics['MAPE'], gb_metrics['MAPE'], lr_metrics['MAPE']]
        })
        st.dataframe(metrics_df)
        best_model = metrics_df.loc[metrics_df['R²'].idxmax(), 'Модель']
        st.success(f"Лучшая модель: {best_model}")

    with tab4:
        st.header("Прогнозирование критического теплового потока")
        col1, col2 = st.columns(2)
        with col1:
            pressure = st.number_input("Давление (МПа)", 0.0, 10.0, 0.5, 0.1)
            mass_flux = st.number_input("Массовый расход (кг/м²·с)", 0.0, 20000.0, 5000.0, 500.0)
            x_e_out = st.number_input("Выходное качество пара", -1.0, 1.0, -0.08, 0.01)
            D_e = st.number_input("Эквивалентный диаметр (мм)", 0.0, 50.0, 3.0, 0.5)
            D_h = st.number_input("Гидравлический диаметр (мм)", 0.0, 50.0, 3.0, 0.5)
            length = st.number_input("Длина канала (мм)", 0.0, 5000.0, 100.0, 50.0)
        with col2:
            st.selectbox("Автор эксперимента", df['author'].unique())
            st.selectbox("Геометрия", ["tube", "plate"])
        
        if st.button("Рассчитать CHF", type="primary"):
            input_df = pd.DataFrame([[pressure, mass_flux, x_e_out, D_e, D_h, length]],
                                   columns=['pressure_[MPa]', 'mass_flux_[kg/m2-s]', 'x_e_out_[-]', 
                                            'D_e_[mm]', 'D_h_[mm]', 'length_[mm]'])
            input_df['pressure_massflux'] = input_df['pressure_[MPa]'] * input_df['mass_flux_[kg/m2-s]']
            input_df['author_encoded'] = 0
            input_df['geometry_encoded'] = 0
            
            input_df = input_df[['pressure_[MPa]', 'mass_flux_[kg/m2-s]', 'x_e_out_[-]', 
                                 'D_e_[mm]', 'D_h_[mm]', 'length_[mm]', 'pressure_massflux',
                                 'author_encoded', 'geometry_encoded']]
            
            model, scaler, _, _, _ = train_random_forest(X, y)
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            
            safety_level, recommendation = get_safety_level(prediction)
            
            st.metric("Прогнозируемый CHF", f"{prediction:.3f} MW/m²")
            st.metric("Уровень безопасности", safety_level)
            
            if safety_level == "CRITICAL":
                st.error("ВНИМАНИЕ: Критический уровень CHF!")
            elif safety_level == "NORMAL":
                st.warning("Уровень CHF в пределах нормы.")
            else:
                st.success("Безопасный уровень CHF.")
            
            st.session_state.history.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'pressure': pressure,
                'mass_flux': mass_flux,
                'predicted_chf': prediction,
                'safety_level': safety_level
            })
            if len(st.session_state.history) > 5:
                st.session_state.history = st.session_state.history[-5:]
        
        st.subheader("История прогнозов (последние 5)")
        if st.session_state.history:
            history_df = pd.DataFrame(st.session_state.history)
            st.dataframe(history_df)
            if st.button("Очистить историю"):
                st.session_state.history = []
                st.rerun()
        else:
            st.info("Нет сохранённых прогнозов.")

if __name__ == "__main__":
    main()
