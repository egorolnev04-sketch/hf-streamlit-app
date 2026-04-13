import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from datetime import datetime
import base64

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
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col])
        label_encoders[col] = le
    
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

def download_plot(fig, filename):
    img_bytes = fig.to_image(format="png")
    b64 = base64.b64encode(img_bytes).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}.png">Скачать PNG</a>'
    return href

def get_safety_level(chf):
    if chf > 2.0:
        return "CRITICAL", "Немедленное внимание", "red"
    elif chf > 1.0:
        return "NORMAL", "Нормальная работа", "orange"
    else:
        return "LOW", "Безопасный режим", "green"

def main():
    st.title("Прогнозирование критического теплового потока (CHF)")
    st.markdown("---")

    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "Random Forest"
    if 'sidebar_filters' not in st.session_state:
        st.session_state.sidebar_filters = {'pressure_range': (0.0, 5.0), 'mass_flux_range': (0.0, 10000.0)}
    if 'log_scale' not in st.session_state:
        st.session_state.log_scale = False

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
            pressure_min = st.session_state.sidebar_filters['pressure_range'][0]
            pressure_max = st.session_state.sidebar_filters['pressure_range'][1]
            pressure_range = st.slider("Давление (МПа)", 0.0, 5.0, (pressure_min, pressure_max), step=0.1, key="pressure_slider")
            
            mass_min = st.session_state.sidebar_filters['mass_flux_range'][0]
            mass_max = st.session_state.sidebar_filters['mass_flux_range'][1]
            mass_flux_range = st.slider("Массовый расход (кг/м²·с)", 0.0, 10000.0, (mass_min, mass_max), step=100.0, key="mass_flux_slider")
            
            st.session_state.sidebar_filters['pressure_range'] = pressure_range
            st.session_state.sidebar_filters['mass_flux_range'] = mass_flux_range
            
            log_scale = st.checkbox("Логарифмическая шкала CHF", value=st.session_state.log_scale)
            st.session_state.log_scale = log_scale
        
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
                st.markdown(download_plot(fig, "3d_scatter"), unsafe_allow_html=True)
                
            elif plot_type == "Heatmap корреляций":
                corr = filtered_df[['pressure_[MPa]', 'mass_flux_[kg/m2-s]', 'x_e_out_[-]', 
                                    'D_e_[mm]', 'D_h_[mm]', 'length_[mm]', 'chf_exp_[MW/m2]']].corr()
                fig = px.imshow(corr, text_auto=True, title="Корреляционная матрица")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(download_plot(fig, "heatmap"), unsafe_allow_html=True)
                
            else:
                filtered_df['pressure_bin'] = pd.cut(filtered_df['pressure_[MPa]'], bins=5)
                fig = px.box(filtered_df, x='pressure_bin', y='chf_exp_[MW/m2]', 
                             title="Распределение CHF по диапазонам давления")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(download_plot(fig, "box_plot"), unsafe_allow_html=True)

    with tab3:
        st.header("Сравнение ML-моделей")
        
        with st.spinner("Обучение моделей..."):
            rf_model, rf_scaler, rf_metrics, rf_y_test, rf_y_pred = train_random_forest(X, y)
            gb_model, gb_scaler, gb_metrics, gb_y_test, gb_y_pred = train_gradient_boosting(X, y)
            lr_model, lr_scaler, lr_metrics, lr_y_test, lr_y_pred = train_linear_regression(X, y)
        
        metrics_df = pd.DataFrame({
            "Модель": ["Random Forest", "Gradient Boosting", "Linear Regression"],
            "R²": [rf_metrics['R2'], gb_metrics['R2'], lr_metrics['R2']],
            "RMSE (MW/m²)": [rf_metrics['RMSE'], gb_metrics['RMSE'], lr_metrics['RMSE']],
            "MAE (MW/m²)": [rf_metrics['MAE'], gb_metrics['MAE'], lr_metrics['MAE']],
            "MAPE (%)": [rf_metrics['MAPE'], gb_metrics['MAPE'], lr_metrics['MAPE']]
        })
        st.dataframe(metrics_df)
        
        col1, col2 = st.columns(2)
        with col1:
            best_model = metrics_df.loc[metrics_df['R²'].idxmax(), 'Модель']
            st.success(f"Лучшая модель: {best_model}")
        
        with col2:
            selected_model_for_vis = st.selectbox("Выберите модель для детального анализа", 
                                                   ["Random Forest", "Gradient Boosting", "Linear Regression"])
        
        if selected_model_for_vis == "Random Forest":
            y_test, y_pred = rf_y_test, rf_y_pred
            model = rf_model
            metrics = rf_metrics
        elif selected_model_for_vis == "Gradient Boosting":
            y_test, y_pred = gb_y_test, gb_y_pred
            model = gb_model
            metrics = gb_metrics
        else:
            y_test, y_pred = lr_y_test, lr_y_pred
            model = lr_model
            metrics = lr_metrics
        
        st.subheader(f"Анализ модели: {selected_model_for_vis}")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel("Фактические CHF (MW/m²)")
        ax.set_ylabel("Предсказанные CHF (MW/m²)")
        ax.set_title("Фактические vs Предсказанные")
        st.pyplot(fig)
        
        st.session_state.selected_model = selected_model_for_vis
        
        if selected_model_for_vis == "Random Forest":
            importances = model.feature_importances_
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.barh(feature_names, importances)
            ax2.set_xlabel("Важность")
            ax2.set_title("Важность признаков")
            st.pyplot(fig2)

    with tab4:
        st.header("Прогнозирование критического теплового потока")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Входные параметры")
            pressure = st.number_input("Давление (МПа)", min_value=0.0, max_value=10.0, value=0.5, step=0.1)
            mass_flux = st.number_input("Массовый расход (кг/м²·с)", min_value=0.0, max_value=20000.0, value=5000.0, step=500.0)
            x_e_out = st.number_input("Выходное качество пара", min_value=-1.0, max_value=1.0, value=-0.08, step=0.01)
            D_e = st.number_input("Эквивалентный диаметр (мм)", min_value=0.0, max_value=50.0, value=3.0, step=0.5)
            D_h = st.number_input("Гидравлический диаметр (мм)", min_value=0.0, max_value=50.0, value=3.0, step=0.5)
            length = st.number_input("Длина канала (мм)", min_value=0.0, max_value=5000.0, value=100.0, step=50.0)
        
        with col2:
            st.subheader("Категориальные параметры")
            author = st.selectbox("Автор эксперимента", df['author'].unique())
            geometry = st.selectbox("Геометрия", ["tube", "plate"])
        
        if pressure <= 0:
            st.warning("Внимание: Давление должно быть положительным!")
        
        if st.button("Рассчитать CHF", type="primary"):
            input_df = pd.DataFrame([[pressure, mass_flux, x_e_out, D_e, D_h, length]],
                                   columns=['pressure_[MPa]', 'mass_flux_[kg/m2-s]', 'x_e_out_[-]', 
                                            'D_e_[mm]', 'D_h_[mm]', 'length_[mm]'])
            input_df['pressure_massflux'] = input_df['pressure_[MPa]'] * input_df['mass_flux_[kg/m2-s]']
            
            if st.session_state.selected_model == "Random Forest":
                model, scaler, _, _, _ = train_random_forest(X, y)
            elif st.session_state.selected_model == "Gradient Boosting":
                model, scaler, _, _, _ = train_gradient_boosting(X, y)
            else:
                model, scaler, _, _, _ = train_linear_regression(X, y)
            
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            
            safety_level, recommendation, color = get_safety_level(prediction)
            
            st.markdown("---")
            st.subheader("Результат прогноза")
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Прогнозируемый CHF", f"{prediction:.3f} MW/m²")
            with col_b:
                st.metric("Уровень безопасности", safety_level)
            with col_c:
                st.metric("Рекомендация", recommendation)
            
            if safety_level == "CRITICAL":
                st.error("ВНИМАНИЕ: Критический уровень CHF! Требуется немедленное вмешательство.")
            elif safety_level == "NORMAL":
                st.warning("Уровень CHF в пределах нормы. Рекомендуется мониторинг.")
            else:
                st.success("Безопасный уровень CHF. Система работает штатно.")
            
            st.session_state.history.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'pressure': pressure,
                'mass_flux': mass_flux,
                'predicted_chf': prediction,
                'safety_level': safety_level
            })
            
            if len(st.session_state.history) > 5:
                st.session_state.history = st.session_state.history[-5:]
        
        st.markdown("---")
        st.subheader("История прогнозов (последние 5)")
        
        if st.session_state.history:
            history_df = pd.DataFrame(st.session_state.history)
            st.dataframe(history_df)
            
            fig_history, ax_history = plt.subplots(figsize=(10, 4))
            ax_history.plot(range(len(history_df)), history_df['predicted_chf'], 'o-')
            ax_history.set_xlabel("Прогноз №")
            ax_history.set_ylabel("CHF (MW/m²)")
            ax_history.set_title("Динамика предсказанных значений CHF")
            st.pyplot(fig_history)
            
            if st.button("Очистить историю"):
                st.session_state.history = []
                st.rerun()
            
            csv = history_df.to_csv(index=False).encode('utf-8')
            st.download_button("Скачать историю (CSV)", csv, "chf_history.csv", "text/csv")
        else:
            st.info("Нет сохранённых прогнозов. Сделайте прогноз, чтобы заполнить историю.")

if __name__ == "__main__":
    main()