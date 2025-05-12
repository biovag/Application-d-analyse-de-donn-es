import streamlit as st
st.set_page_config(page_title="Analyse de Données", layout="wide")
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.api as sm
from patsy import dmatrices
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
import io

st.title("📊 Application d'Analyse de Données")
st.subheader("Auteur : Biova Gatien DANGNON")

# === Étape 1 : Upload du fichier ===
uploaded_file = st.file_uploader("Téléchargez un fichier CSV ou Excel", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.sidebar.title("📁 Menu")
    option = st.sidebar.radio(
        "Choisissez un module :",
        (
            "Aperçu des données",
            "Statistiques descriptives",
            "Visualisation",
            "Tests statistiques",
            "Modèles prédictifs",
            "Analyse textuelle",
            "Clustering / ACP",
            "Séries temporelles",
            "GLM & GLMM",
            "Exporter rapport"
        )
    )

    if option == "Aperçu des données":
        st.subheader("🔍 Aperçu des données")
        st.write(df.head())

    elif option == "Statistiques descriptives":
        st.subheader("📈 Statistiques descriptives")
        st.write(df.describe())

    elif option == "Visualisation":
        st.subheader("📊 Visualisation de données")
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if len(numeric_cols) >= 1:
            col = st.selectbox("Choisissez une variable numérique à visualiser", numeric_cols)
            chart_type = st.radio("Type de graphique :", ["Histogramme", "Boxplot", "Nuage de points (2 variables)"])

            if chart_type == "Histogramme":
                fig = px.histogram(df, x=col, nbins=30, title=f"Distribution de {col}")
                st.plotly_chart(fig)
            elif chart_type == "Boxplot":
                fig = px.box(df, y=col, title=f"Boxplot de {col}")
                st.plotly_chart(fig)
            elif chart_type == "Nuage de points (2 variables)":
                col2 = st.selectbox("Choisissez une deuxième variable", numeric_cols)
                fig = px.scatter(df, x=col, y=col2, trendline="ols")
                st.plotly_chart(fig)
        else:
            st.warning("Aucune variable numérique détectée.")

    elif option == "Tests statistiques":
        st.subheader("📊 Tests Statistiques")
        from scipy import stats
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if len(numeric_cols) >= 2:
            var1 = st.selectbox("Variable 1", numeric_cols)
            var2 = st.selectbox("Variable 2", numeric_cols, index=1)
            test = st.radio("Test à effectuer", ["Test t", "Corrélation Pearson"])
            if st.button("Lancer le test"):
                if test == "Test t":
                    stat, p = stats.ttest_ind(df[var1].dropna(), df[var2].dropna())
                    st.write(f"Test t entre {var1} et {var2} : t={stat:.2f}, p={p:.4f}")
                else:
                    corr, p = stats.pearsonr(df[var1].dropna(), df[var2].dropna())
                    st.write(f"Corrélation de Pearson : r={corr:.2f}, p={p:.4f}")

    elif option == "Modèles prédictifs":
        st.subheader("🤖 Modèles Prédictifs")
        target = st.selectbox("Variable cible", df.columns)
        features = st.multiselect("Variables explicatives", [col for col in df.columns if col != target])
        model_type = st.radio("Type de modèle", ["Régression linéaire", "Régression logistique"])
        if st.button("Entraîner le modèle"):
            X = df[features]
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            if model_type == "Régression linéaire":
                model = LinearRegression()
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                st.write(f"MSE : {mean_squared_error(y_test, pred):.2f}")
            else:
                model = LogisticRegression()
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                st.write("Rapport de classification :")
                st.text(classification_report(y_test, pred))

    elif option == "Analyse textuelle":
        st.subheader("📝 Analyse textuelle")
        text_col = st.selectbox("Choisissez une colonne textuelle", df.select_dtypes(include='object').columns)
        vec = CountVectorizer(stop_words='french')
        X = vec.fit_transform(df[text_col].fillna(""))
        word_counts = pd.DataFrame(X.toarray(), columns=vec.get_feature_names_out()).sum().sort_values(ascending=False)
        st.write("Nuage de mots :")
        st.bar_chart(word_counts.head(20))

    elif option == "Clustering / ACP":
        st.subheader("📌 Clustering et ACP")
        numeric_cols = df.select_dtypes(include='number')
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(numeric_cols)
        df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
        kmeans = KMeans(n_clusters=3)
        df_pca['Cluster'] = kmeans.fit_predict(numeric_cols)
        fig = px.scatter(df_pca, x="PC1", y="PC2", color=df_pca['Cluster'].astype(str))
        st.plotly_chart(fig)

    elif option == "Séries temporelles":
        st.subheader("📈 Analyse de séries temporelles")
        date_col = st.selectbox("Colonne date", df.select_dtypes(include='datetime64').columns)
        value_col = st.selectbox("Valeur à tracer", df.select_dtypes(include='number').columns)
        df_sorted = df.sort_values(by=date_col)
        fig = px.line(df_sorted, x=date_col, y=value_col)
        st.plotly_chart(fig)

    elif option == "GLM & GLMM":
        st.subheader("📌 Modèles Linéaires Généralisés (GLM) et Mixtes (GLMM)")
        st.markdown("### 1. Formule du modèle (type R)")
        st.code("exemple : y ~ x1 + x2", language='r')
        formula = st.text_input("Saisissez la formule :", value="")
        family_name = st.selectbox("Famille", ["Gaussian", "Binomial", "Poisson"])
        if st.button("Exécuter le modèle GLM"):
            try:
                y, X = dmatrices(formula, data=df, return_type='dataframe')
                family = getattr(sm.families, family_name)()
                model = sm.GLM(y, X, family=family)
                results = model.fit()
                st.markdown("### Résumé du modèle")
                st.text(results.summary())
            except Exception as e:
                st.error(f"Erreur : {e}")

        st.markdown("---")
        st.markdown("📌 **Modèles Linéaires Mixtes (GLMM)** *(bêta)*")
        st.warning("GLMM avec effets aléatoires nécessite l'utilisation de `mixedlm` (limité à la famille Gaussienne)")
        formula_glmm = st.text_input("Formule GLMM (ex: y ~ x1)", key="glmm_formula")
        group_var = st.selectbox("Variable de groupe (effet aléatoire)", df.columns)
        if st.button("Exécuter le modèle GLMM"):
            try:
                model = sm.MixedLM.from_formula(formula_glmm, groups=df[group_var], data=df)
                result = model.fit()
                st.markdown("### Résumé GLMM")
                st.text(result.summary())
            except Exception as e:
                st.error(f"Erreur : {e}")

    elif option == "Exporter rapport":
        st.subheader("📁 Exporter le rapport")
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        st.download_button("📥 Télécharger les données", buffer.getvalue(), file_name="rapport_analyse.csv", mime="text/csv")
else:
    st.info("Veuillez uploader un fichier pour commencer.")
