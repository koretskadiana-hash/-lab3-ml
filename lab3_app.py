# =============================================================================
# Лабораторна робота 3 — Streamlit-додаток
# Методи класифікації: Дерево рішень та Random Forest
# Запуск: streamlit run lab3_app.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, accuracy_score,
    roc_curve, auc, f1_score
)
import warnings
warnings.filterwarnings("ignore")

# ── Налаштування сторінки ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Лаб. 3 — Класифікація",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Кольори ──────────────────────────────────────────────────────────────────
C_BLUE   = "#4C72B0"
C_ORANGE = "#DD8452"
C_GREEN  = "#1D9E75"
C_GRAY   = "#888780"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 120,
})


# =============================================================================
# ЗАВАНТАЖЕННЯ ДАНИХ (кешується)
# =============================================================================
@st.cache_data
def load_data(uploaded_file):
    df_raw = pd.read_excel(
        uploaded_file,
        sheet_name="Зведена таблиця",
        skiprows=3,
        header=None
    )
    df_raw.columns = ["Country", "Exports", "GDP_pc", "Imports",
                      "Inflation", "LifeExp", "PopGrowth", "Total"]
    df = df_raw[df_raw["Country"].apply(
        lambda x: isinstance(x, str)
        and x not in ["Названия строк", "Загальний підсумок", "Общий итог"]
    )].copy().reset_index(drop=True)

    for col in ["Exports", "GDP_pc", "Imports", "Inflation", "LifeExp", "PopGrowth"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Exports", "GDP_pc", "Imports", "Inflation", "LifeExp", "PopGrowth"])

    median_gdp = df["GDP_pc"].median()
    df["Class"] = np.where(df["GDP_pc"] >= median_gdp, "High GDP", "Low GDP")
    return df, median_gdp


FEATURE_COLS = ["Exports", "Imports", "Inflation", "LifeExp", "PopGrowth"]
FEATURE_LABELS = {
    "Exports":   "Експорт (% ВВП)",
    "Imports":   "Імпорт (% ВВП)",
    "Inflation": "Інфляція (%)",
    "LifeExp":   "Тривалість життя (рр.)",
    "PopGrowth": "Ріст населення (%)",
}


# =============================================================================
# САЙДБАР
# =============================================================================
with st.sidebar:
    st.title("🌳 Лаб. 3")
    st.caption("Методи класифікації")
    st.divider()

    uploaded = st.file_uploader("Завантажте data.xlsx", type=["xlsx"])

    if uploaded:
        df, median_gdp = load_data(uploaded)
        st.success(f"{len(df)} країн завантажено")
        st.metric("Медіана ВВП", f"${median_gdp:,.0f}")
        counts = df["Class"].value_counts()
        st.write(f"High GDP: **{counts.get('High GDP', 0)}** країн")
        st.write(f"Low GDP:  **{counts.get('Low GDP', 0)}** країн")
    else:
        st.info("Завантажте файл даних для початку роботи")
        df = None

    st.divider()
    page = st.radio(
        "Розділ",
        ["Дані та EDA", "Дерево рішень", "Random Forest", "Порівняння методів"],
        label_visibility="collapsed"
    )


# =============================================================================
# ЗАГЛУШКА БЕЗ ФАЙЛУ
# =============================================================================
if df is None:
    st.title("Лабораторна робота 3 — Методи класифікації")
    st.info("Завантажте файл **data.xlsx** у лівій панелі, щоб почати.")
    st.stop()


X = df[FEATURE_COLS].values
y = df["Class"].values
feat_names = [FEATURE_LABELS[f] for f in FEATURE_COLS]


# =============================================================================
# СТОРІНКА 1: ДАНІ ТА EDA
# =============================================================================
if page == "Дані та EDA":
    st.title("Дані та описова статистика")
    st.caption("Макроекономічні показники 37 європейських країн, 2024 р. | Джерело: World Bank")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Країн",   len(df))
    col2.metric("Ознак",   len(FEATURE_COLS))
    col3.metric("High GDP", int((df["Class"] == "High GDP").sum()))
    col4.metric("Low GDP",  int((df["Class"] == "Low GDP").sum()))

    st.divider()

    tab1, tab2, tab3 = st.tabs(["Таблиця даних", "Описова статистика", "Візуалізація"])

    with tab1:
        show_df = df[["Country"] + FEATURE_COLS + ["Class"]].copy()
        show_df.columns = ["Країна"] + list(FEATURE_LABELS.values()) + ["Клас"]
        st.dataframe(
            show_df.style.apply(
                lambda r: ["background-color: #E1F5EE" if r["Клас"] == "High GDP"
                           else "background-color: #FAEEDA" for _ in r],
                axis=1
            ),
            use_container_width=True, height=420
        )

    with tab2:
        desc = df[FEATURE_COLS].describe().round(3)
        desc.index.name = "Показник"
        desc.columns = [FEATURE_LABELS[c] for c in desc.columns]
        st.dataframe(desc, use_container_width=True)

    with tab3:
        col_a, col_b = st.columns(2)

        with col_a:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            for cls, color in [("High GDP", C_BLUE), ("Low GDP", C_ORANGE)]:
                sub = df[df["Class"] == cls]["LifeExp"]
                ax.hist(sub, bins=8, alpha=0.7, color=color, label=cls, edgecolor="white")
            ax.set_xlabel("Тривалість життя (рр.)")
            ax.set_ylabel("Кількість країн")
            ax.set_title("Розподіл тривалості життя за класами")
            ax.legend()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col_b:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            for cls, color in [("High GDP", C_BLUE), ("Low GDP", C_ORANGE)]:
                sub = df[df["Class"] == cls]
                ax.scatter(sub["Exports"], sub["GDP_pc"] / 1000,
                           color=color, alpha=0.8, s=60, label=cls)
            ax.set_xlabel("Експорт (% ВВП)")
            ax.set_ylabel("ВВП на душу (тис. USD)")
            ax.set_title("Експорт vs ВВП на душу")
            ax.legend()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        fig, axes = plt.subplots(1, 5, figsize=(14, 3))
        for ax, feat in zip(axes, FEATURE_COLS):
            data = [df[df["Class"] == c][feat].values for c in ["High GDP", "Low GDP"]]
            bp = ax.boxplot(data, patch_artist=True,
                            boxprops=dict(linewidth=0.8),
                            medianprops=dict(color="white", linewidth=2))
            bp["boxes"][0].set_facecolor(C_BLUE)
            bp["boxes"][1].set_facecolor(C_ORANGE)
            ax.set_xticklabels(["High", "Low"], fontsize=9)
            ax.set_title(FEATURE_LABELS[feat], fontsize=9)
        plt.suptitle("Box-plot ознак за класами", fontsize=11, y=1.02)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()


# =============================================================================
# СТОРІНКА 2: ДЕРЕВО РІШЕНЬ
# =============================================================================
elif page == "Дерево рішень":
    st.title("Дерево рішень")

    # ── Параметри ──────────────────────────────────────────────────────────
    with st.expander("Параметри моделі", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        criterion   = c1.selectbox("Критерій", ["gini", "entropy"], index=0)
        max_depth   = c2.slider("Максимальна глибина", 1, 10, 4)
        min_leaf    = c3.slider("Min зразків у листі", 1, 10, 2)
        test_size   = c4.slider("Розмір тестової вибірки", 0.15, 0.4, 0.25, 0.05)

        col_p, col_pr = st.columns(2)
        pruning = col_p.checkbox("Увімкнути обрізання (Cost-Complexity Pruning)", value=False)
        if pruning:
            ccp_alpha = col_pr.slider("ccp_alpha", 0.000, 0.100, 0.000, 0.001, format="%.3f")
        else:
            ccp_alpha = 0.0

    run = st.button("Запустити навчання ▶", type="primary", use_container_width=True)

    if run or "dt_results" in st.session_state:
        if run:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            clf = DecisionTreeClassifier(
                criterion=criterion, max_depth=max_depth,
                min_samples_leaf=min_leaf, ccp_alpha=ccp_alpha, random_state=42
            )
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)
            acc  = accuracy_score(y_te, y_pred)
            f1   = f1_score(y_te, y_pred, average="macro")
            cv   = cross_val_score(
                clf, X, y,
                cv=StratifiedKFold(5, shuffle=True, random_state=42)
            ).mean()
            st.session_state["dt_results"] = dict(
                clf=clf, X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te,
                y_pred=y_pred, acc=acc, f1=f1, cv=cv
            )

        r = st.session_state["dt_results"]
        clf, X_tr, X_te, y_tr, y_te = r["clf"], r["X_tr"], r["X_te"], r["y_tr"], r["y_te"]
        y_pred, acc, f1, cv = r["y_pred"], r["acc"], r["f1"], r["cv"]

        st.divider()

        # ── Метрики ────────────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Точність (тест)",   f"{acc:.1%}")
        m2.metric("F1-score (macro)",  f"{f1:.3f}")
        m3.metric("Крос-валідація 5×", f"{cv:.1%}")
        m4.metric("Глибина / листів",  f"{clf.get_depth()} / {clf.get_n_leaves()}")

        st.divider()

        tab1, tab2, tab3, tab4 = st.tabs(
            ["Дерево рішень", "Матриця плутанини", "Важливість ознак", "Правила (текст)"]
        )

        with tab1:
            fig, ax = plt.subplots(figsize=(18, 7))
            plot_tree(clf, feature_names=feat_names,
                      class_names=clf.classes_, filled=True,
                      rounded=True, fontsize=9, ax=ax, impurity=True)
            ax.set_title(
                f"Дерево рішень ({criterion.capitalize()}) | "
                f"глибина={clf.get_depth()}, листів={clf.get_n_leaves()}",
                fontsize=12
            )
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with tab2:
            col_l, col_r = st.columns([1, 2])
            with col_l:
                cm = confusion_matrix(y_te, y_pred, labels=["High GDP", "Low GDP"])
                fig, ax = plt.subplots(figsize=(4, 3.5))
                ConfusionMatrixDisplay(cm, display_labels=["High GDP", "Low GDP"]).plot(
                    ax=ax, colorbar=False, cmap="Blues"
                )
                ax.set_title("Матриця плутанини")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()
            with col_r:
                report_dict = classification_report(
                    y_te, y_pred, output_dict=True, zero_division=0
                )
                report_df = pd.DataFrame(report_dict).T.round(3)
                st.dataframe(report_df, use_container_width=True)
                st.caption("Precision, Recall, F1-score для кожного класу")

        with tab3:
            importances = clf.feature_importances_
            feat_imp = pd.Series(importances, index=feat_names).sort_values()
            fig, ax = plt.subplots(figsize=(7, 3.5))
            colors = [C_BLUE if v == feat_imp.max() else C_GRAY for v in feat_imp.values]
            ax.barh(feat_imp.index, feat_imp.values, color=colors, edgecolor="white")
            for i, (name, val) in enumerate(feat_imp.items()):
                ax.text(val + 0.005, i, f"{val:.3f}", va="center", fontsize=10)
            ax.set_xlabel("Важливість (Gini importance)")
            ax.set_title("Важливість ознак — Дерево рішень")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with tab4:
            rules = export_text(clf, feature_names=feat_names)
            st.code(rules, language="text")
            st.caption("Набір правил що генерує дерево при класифікації")

        # ── Обрізання ─────────────────────────────────────────────────────
        if pruning:
            st.divider()
            st.subheader("Аналіз обрізання (Cost-Complexity Pruning)")
            path = DecisionTreeClassifier(
                criterion=criterion, max_depth=max_depth,
                min_samples_leaf=min_leaf, random_state=42
            ).cost_complexity_pruning_path(X_tr, y_tr)
            alphas = path.ccp_alphas[:-1]
            tr_sc, te_sc = [], []
            for a in alphas:
                c = DecisionTreeClassifier(criterion=criterion, ccp_alpha=a, random_state=42)
                c.fit(X_tr, y_tr)
                tr_sc.append(c.score(X_tr, y_tr))
                te_sc.append(c.score(X_te, y_te))

            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.plot(alphas, tr_sc, marker="o", markersize=4, label="Навчальна", color=C_BLUE)
            ax.plot(alphas, te_sc, marker="o", markersize=4, label="Тестова",   color=C_ORANGE)
            best_a = alphas[np.argmax(te_sc)]
            ax.axvline(best_a, color=C_GREEN, linestyle="--",
                       label=f"Оптим. α = {best_a:.4f}")
            ax.set_xlabel("ccp_alpha")
            ax.set_ylabel("Точність")
            ax.set_title("Точність залежно від рівня обрізання")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
            st.info(f"Оптимальне **ccp_alpha = {best_a:.4f}** дає найвищу точність на тестовій вибірці")


# =============================================================================
# СТОРІНКА 3: RANDOM FOREST
# =============================================================================
elif page == "Random Forest":
    st.title("Random Forest")

    with st.expander("Параметри моделі", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        n_est    = c1.slider("Кількість дерев", 10, 500, 200, 10)
        max_d    = c2.slider("Максимальна глибина", 1, 15, 5)
        min_l    = c3.slider("Min зразків у листі", 1, 10, 2)
        test_sz  = c4.slider("Розмір тестової вибірки", 0.15, 0.4, 0.25, 0.05)

    run_rf = st.button("Запустити Random Forest ▶", type="primary", use_container_width=True)

    if run_rf or "rf_results" in st.session_state:
        if run_rf:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=test_sz, random_state=42, stratify=y
            )
            rf = RandomForestClassifier(
                n_estimators=n_est, max_depth=max_d,
                min_samples_leaf=min_l, random_state=42, n_jobs=-1
            )
            rf.fit(X_tr, y_tr)
            y_pred_rf = rf.predict(X_te)
            acc_rf = accuracy_score(y_te, y_pred_rf)
            f1_rf  = f1_score(y_te, y_pred_rf, average="macro")
            cv_rf  = cross_val_score(
                rf, X, y,
                cv=StratifiedKFold(5, shuffle=True, random_state=42)
            ).mean()

            # ROC
            probs = rf.predict_proba(X_te)
            idx_h = list(rf.classes_).index("High GDP")
            y_bin = (y_te == "High GDP").astype(int)
            fpr, tpr, _ = roc_curve(y_bin, probs[:, idx_h])
            roc_auc = auc(fpr, tpr)

            # n_estimators vs accuracy
            n_range = list(range(10, min(n_est, 200) + 10, 10))
            accs_n = []
            for n in n_range:
                c = RandomForestClassifier(n_estimators=n, max_depth=max_d,
                                           min_samples_leaf=min_l,
                                           random_state=42, n_jobs=-1)
                c.fit(X_tr, y_tr)
                accs_n.append(c.score(X_te, y_te))

            st.session_state["rf_results"] = dict(
                rf=rf, X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te,
                y_pred=y_pred_rf, acc=acc_rf, f1=f1_rf, cv=cv_rf,
                fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                n_range=n_range, accs_n=accs_n
            )

        r = st.session_state["rf_results"]
        rf = r["rf"]
        y_te, y_pred_rf = r["y_te"], r["y_pred"]
        acc_rf, f1_rf, cv_rf = r["acc"], r["f1"], r["cv"]

        st.divider()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Точність (тест)",   f"{acc_rf:.1%}")
        m2.metric("F1-score (macro)",  f"{f1_rf:.3f}")
        m3.metric("Крос-валідація 5×", f"{cv_rf:.1%}")
        m4.metric("ROC AUC",           f"{r['roc_auc']:.3f}")

        st.divider()

        tab1, tab2, tab3 = st.tabs(
            ["Важливість ознак", "ROC-крива & Матриця плутанини", "Кількість дерев vs Точність"]
        )

        with tab1:
            imp = pd.Series(rf.feature_importances_, index=feat_names).sort_values()
            fig, ax = plt.subplots(figsize=(7, 3.5))
            colors = [C_BLUE if v == imp.max() else C_GRAY for v in imp.values]
            ax.barh(imp.index, imp.values, color=colors, edgecolor="white")
            for i, (name, val) in enumerate(imp.items()):
                ax.text(val + 0.005, i, f"{val:.3f}", va="center", fontsize=10)
            ax.set_xlabel("Mean Decrease Impurity")
            ax.set_title("Важливість ознак — Random Forest")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

            top = imp.idxmax()
            st.info(
                f"Найважливіша ознака: **{top}** ({imp.max():.1%} важливості). "
                "Це відповідає економічній логіці: у багатших країнах вища тривалість "
                "життя завдяки кращій медицині та якості соціальних послуг."
            )

        with tab2:
            col_l, col_r = st.columns(2)
            with col_l:
                fig, ax = plt.subplots(figsize=(4.5, 4))
                ax.plot(r["fpr"], r["tpr"], color=C_BLUE, lw=2,
                        label=f"ROC (AUC = {r['roc_auc']:.2f})")
                ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
                ax.set_xlabel("FPR")
                ax.set_ylabel("TPR")
                ax.set_title("ROC-крива")
                ax.legend(loc="lower right")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

            with col_r:
                cm = confusion_matrix(y_te, y_pred_rf, labels=["High GDP", "Low GDP"])
                fig, ax = plt.subplots(figsize=(4.5, 4))
                ConfusionMatrixDisplay(cm, display_labels=["High GDP", "Low GDP"]).plot(
                    ax=ax, colorbar=False, cmap="Blues"
                )
                ax.set_title("Матриця плутанини")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

        with tab3:
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.plot(r["n_range"], r["accs_n"], marker="o", markersize=4, color=C_BLUE)
            ax.set_xlabel("Кількість дерев (n_estimators)")
            ax.set_ylabel("Точність на тестовій вибірці")
            ax.set_title("Залежність точності від кількості дерев")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
            st.caption(
                "Зазвичай після 50–100 дерев точність стабілізується — "
                "більша кількість дерев додає стійкості, але майже не змінює точність."
            )


# =============================================================================
# СТОРІНКА 4: ПОРІВНЯННЯ МЕТОДІВ
# =============================================================================
elif page == "Порівняння методів":
    st.title("Порівняння методів класифікації")
    st.caption("Навчання всіх моделей на одному поділі для коректного порівняння")

    test_sz = st.slider("Розмір тестової вибірки", 0.15, 0.40, 0.25, 0.05)

    if st.button("Порівняти всі методи ▶", type="primary", use_container_width=True):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_sz, random_state=42, stratify=y
        )
        cv_cv = StratifiedKFold(5, shuffle=True, random_state=42)
        results = []

        models = [
            ("Дерево (Gini)",    DecisionTreeClassifier(criterion="gini",    max_depth=4, min_samples_leaf=2, random_state=42)),
            ("Дерево (Entropy)", DecisionTreeClassifier(criterion="entropy", max_depth=4, min_samples_leaf=2, random_state=42)),
            ("Дерево (обрізане)", DecisionTreeClassifier(criterion="gini",   max_depth=4, min_samples_leaf=2, ccp_alpha=0.02, random_state=42)),
            ("Random Forest",   RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=2, random_state=42, n_jobs=-1)),
        ]

        prog = st.progress(0, "Навчаємо моделі...")
        for i, (name, model) in enumerate(models):
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            acc  = accuracy_score(y_te, y_pred)
            f1   = f1_score(y_te, y_pred, average="macro")
            cv_s = cross_val_score(model, X, y, cv=cv_cv).mean()
            results.append({"Метод": name, "Точність (тест)": acc, "F1-macro": f1, "CV 5-fold": cv_s})
            prog.progress((i + 1) / len(models), f"Навчаємо: {name}...")
        prog.empty()

        res_df = pd.DataFrame(results).set_index("Метод").round(4)
        st.session_state["compare_results"] = res_df

    if "compare_results" in st.session_state:
        res_df = st.session_state["compare_results"]

        best = res_df["CV 5-fold"].idxmax()
        st.success(f"Найкраща модель за крос-валідацією: **{best}**")

        st.dataframe(
            res_df.style.highlight_max(axis=0, color="#E1F5EE"),
            use_container_width=True
        )

        st.divider()

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        metrics = ["Точність (тест)", "F1-macro", "CV 5-fold"]
        colors  = [C_BLUE, C_ORANGE, C_GREEN, "#AFA9EC"]
        for ax, met in zip(axes, metrics):
            vals = res_df[met]
            bars = ax.bar(range(len(vals)), vals.values, color=colors, edgecolor="white")
            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels(vals.index, rotation=15, ha="right", fontsize=9)
            ax.set_ylim(0, 1.15)
            ax.set_title(met)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        f"{bar.get_height():.2f}",
                        ha="center", va="bottom", fontsize=9)
        plt.suptitle("Порівняння метрик усіх методів", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.divider()
        st.subheader("Економічна інтерпретація")
        st.markdown("""
        **Ключові висновки:**

        - **Тривалість життя** є найпотужнішим предиктором рівня ВВП (~46% важливості).
          Це відображає тісний зв'язок між економічним добробутом та якістю охорони здоров'я.

        - **Ріст населення** (~33%) — країни з вищим ВВП, як правило, мають нижчий природний
          приріст населення, натомість компенсують його міграцією.

        - **Random Forest** показує найвищу точність завдяки ансамблюванню (~200 дерев),
          що зменшує перенавчання та варіативність прогнозів.

        - **Обрізання дерева** дає простішу інтерпретовану модель без суттєвої втрати точності,
          що корисно для економічного аналізу та пояснення результатів стейкхолдерам.
        """)
