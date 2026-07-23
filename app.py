from datetime import datetime
import os
import streamlit as st
from src.chain import ask
from src.ingest import build_vectorstore


@st.cache_resource
def charger_vector_store():
    return build_vectorstore()


# ============================================================
# Configuration de la page
# ============================================================
st.set_page_config(
    page_title="PolicyBot | Assistant Documentaire",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# "Base" d'utilisateurs codée en dur
# ============================================================
USERS = {
    "admin": {"password": "admin123", "role": "admin", "nom": "Administrateur"},
    "alice": {"password": "alice123", "role": "user", "nom": "Alice Martin"},
    "bob": {"password": "bob123", "role": "user", "nom": "Bob Dupont"},
}

# ============================================================
# Initialisation de l'état
# ============================================================
if "authentifie" not in st.session_state:
    st.session_state.authentifie = False
if "username" not in st.session_state:
    st.session_state.username = None
if "role" not in st.session_state:
    st.session_state.role = None

if "historique" not in st.session_state:
    st.session_state.historique = []
if "historique_global" not in st.session_state:
    st.session_state.historique_global = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode_sombre" not in st.session_state:
    st.session_state.mode_sombre = False


# ============================================================
# Palette & styles globaux
# ============================================================
if st.session_state.mode_sombre:
    BG = "#0b0f19"
    BG_CARD = "#141a26"
    BG_CARD_HOVER = "#1a2233"
    TEXT = "#f5f7fa"
    TEXT_MUTED = "#9aa4b2"
    BORDER = "#232b3a"
    ACCENT = "#5b8def"
    ACCENT_SOFT = "rgba(91,141,239,0.15)"
else:
    BG = "#f6f8fb"
    BG_CARD = "#ffffff"
    BG_CARD_HOVER = "#f0f4fa"
    TEXT = "#101828"
    TEXT_MUTED = "#667085"
    BORDER = "#e4e7ec"
    ACCENT = "#2f5fe0"
    ACCENT_SOFT = "rgba(47,95,224,0.10)"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {BG};
        color: {TEXT};
    }}
    header[data-testid="stHeader"] {{ background: transparent; }}
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    .topbar {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 14px 22px;
        background: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 14px;
        margin-bottom: 18px;
        box-shadow: 0 1px 3px rgba(16,24,40,0.06);
    }}
    .brand {{ display: flex; align-items: center; gap: 12px; }}
    .brand-icon {{
        width: 42px; height: 42px; border-radius: 10px;
        background: {ACCENT}; display: flex; align-items: center; justify-content: center;
        font-size: 20px;
    }}
    .brand-title {{ font-size: 19px; font-weight: 700; color: {TEXT}; line-height: 1.1; }}
    .brand-sub {{ font-size: 12.5px; color: {TEXT_MUTED}; }}

    .badge-role {{
        font-size: 11.5px;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 20px;
        background: {ACCENT_SOFT};
        color: {ACCENT};
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}

    .doc-card {{
        background: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 10px 12px;
        margin-bottom: 8px;
        font-size: 13.5px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: background 0.15s ease;
    }}
    .doc-card:hover {{ background: {BG_CARD_HOVER}; }}

    .hist-item {{
        background: {BG_CARD};
        border-left: 3px solid {ACCENT};
        border-radius: 6px;
        padding: 8px 10px;
        margin-bottom: 6px;
        font-size: 13px;
        color: {TEXT};
    }}

    .answer-box {{
        background: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 20px 22px;
        margin-top: 10px;
        box-shadow: 0 1px 3px rgba(16,24,40,0.05);
    }}
    .answer-label {{
        font-size: 12px; letter-spacing: 0.06em; text-transform: uppercase;
        color: {TEXT_MUTED}; font-weight: 600; margin-bottom: 6px;
    }}

    .metric-card {{
        background: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 16px 18px;
        text-align: left;
    }}
    .metric-value {{ font-size: 24px; font-weight: 700; color: {TEXT}; }}
    .metric-label {{ font-size: 12.5px; color: {TEXT_MUTED}; margin-top: 2px; }}

    .section-title {{
        font-size: 13px; font-weight: 700; letter-spacing: 0.04em;
        text-transform: uppercase; color: {TEXT_MUTED}; margin: 4px 0 10px 0;
    }}

    .login-card {{
        max-width: 380px;
        margin: 60px auto;
        background: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 32px 30px;
        box-shadow: 0 4px 16px rgba(16,24,40,0.08);
        text-align: center;
    }}

    div[data-testid="stTextInput"] input {{
        background: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 15px;
        color: {TEXT};
    }}

    div.stButton > button {{
        border-radius: 10px;
        border: 1px solid {BORDER};
        background: {BG_CARD};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PAGE DE LOGIN
# ============================================================
def afficher_login():
    st.markdown(
        f"""
        <div class="login-card">
            <div class="brand-icon" style="margin: 0 auto 14px auto;">📄</div>
            <div class="brand-title" style="font-size:22px;">PolicyBot</div>
            <div class="brand-sub" style="margin-bottom: 18px;">Connecte-toi pour accéder à l'assistant</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_vide1, col_form, col_vide2 = st.columns([1, 1.2, 1])
    with col_form:
        with st.form("form_login"):
            identifiant = st.text_input("Identifiant")
            mot_de_passe = st.text_input("Mot de passe", type="password")
            valider = st.form_submit_button("Se connecter", use_container_width=True)

        if valider:
            utilisateur = USERS.get(identifiant)
            if utilisateur and utilisateur["password"] == mot_de_passe:
                st.session_state.authentifie = True
                st.session_state.username = identifiant
                st.session_state.role = utilisateur["role"]
                st.rerun()
            else:
                st.error("Identifiant ou mot de passe incorrect.")

        with st.expander("Comptes de démonstration"):
            st.caption("**admin** / admin123 (rôle admin)")
            st.caption("**alice** / alice123 (rôle utilisateur)")
            st.caption("**bob** / bob123 (rôle utilisateur)")


# ============================================================
# BANDEAU SUPÉRIEUR (commun)
# ============================================================
def afficher_topbar():
    maintenant = datetime.now().strftime("%d/%m/%Y — %H:%M:%S")
    st.markdown(
        f'<div style="text-align:right; font-size:13px; color:{TEXT_MUTED}; '
        f'padding: 2px 4px 10px 4px;">🕐 {maintenant}</div>',
        unsafe_allow_html=True,
    )

    col_brand, col_user, col_theme, col_logout = st.columns([6, 2, 1, 1])

    with col_brand:
        role_label = (
            "Admin" if st.session_state.role == "admin" else "Utilisateur"
        )
        st.markdown(
            f"""
            <div class="topbar" style="justify-content:flex-start; gap:14px;">
                <div class="brand">
                    <div class="brand-icon">📄</div>
                    <div>
                        <div class="brand-title">PolicyBot</div>
                        <div class="brand-sub">Assistant documentaire interne</div>
                    </div>
                </div>
                <span class="badge-role">{role_label}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_user:
        nom = USERS[st.session_state.username]["nom"]
        st.markdown(
            f"""
            <div class="topbar" style="justify-content:center; height:100%;">
                <div style="font-size:13px; color:{TEXT};">👤 {nom}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_theme:
        icone = "🌙" if not st.session_state.mode_sombre else "☀️"
        if st.button(icone, key="toggle_theme"):
            st.session_state.mode_sombre = not st.session_state.mode_sombre
            st.rerun()

    with col_logout:
        if st.button("🚪", key="logout", help="Se déconnecter"):
            st.session_state.authentifie = False
            st.session_state.username = None
            st.session_state.role = None
            st.rerun()

    st.divider()



# ============================================================
# DASHBOARD UTILISATEUR
# ============================================================
def lire_apercu_document(chemin_fichier, max_caracteres=2000):
    """Lit un aperçu du contenu d'un document (txt ou pdf)."""
    try:
        if chemin_fichier.endswith(".txt"):
            with open(chemin_fichier, "r", encoding="utf-8") as f:
                contenu = f.read()
            return contenu[:max_caracteres]
        elif chemin_fichier.endswith(".pdf"):
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(chemin_fichier)
            pages = loader.load()
            contenu = "\n\n".join(p.page_content for p in pages)
            return contenu[:max_caracteres]
        else:
            return "Format non pris en charge pour l'aperçu."
    except Exception as e:
        return f"Erreur lors de la lecture : {e}"

def afficher_dashboard_utilisateur():
    with st.sidebar:
        st.markdown('<div class="section-title">📁 Documents disponibles</div>', unsafe_allow_html=True)
        dossier_data = "data/raw"
        if os.path.exists(dossier_data):
            fichiers = sorted(os.listdir(dossier_data))
            if fichiers:
                for f in fichiers:
                    with st.expander(f"📄 {f}"):
                        apercu = lire_apercu_document(os.path.join(dossier_data, f))
                        st.text(apercu + "...")
            else:
                st.caption("Dossier vide.")
        else:
            st.caption("Aucun document trouvé.")

    nb_docs = len(os.listdir("data/raw")) if os.path.exists("data/raw") else 0
    nb_questions = len([m for m in st.session_state.messages if m["role"] == "user"])

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{nb_docs}</div>'
            f'<div class="metric-label">Documents indexés</div></div>',
            unsafe_allow_html=True,
        )
    with col_m2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{nb_questions}</div>'
            f'<div class="metric-label">Mes questions</div></div>',
            unsafe_allow_html=True,
        )
    with col_m3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value" style="font-size:18px;">🟢 En ligne</div>'
            f'<div class="metric-label">Statut du service</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")

    # ===== Historique de conversation =====
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("📄 Sources"):
                    for s in message["sources"]:
                        st.caption(s)

    # ===== Nouveau message =====
    question = st.chat_input("Pose ta question sur les documents...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        vector_store = charger_vector_store()
        with st.chat_message("assistant"):
            with st.spinner("Recherche en cours..."):
                reponse, sources = ask(question, vector_store)
            st.markdown(reponse)
            if sources:
                with st.expander("📄 Sources"):
                    for s in sources:
                        st.caption(s)

        st.session_state.messages.append(
            {"role": "assistant", "content": reponse, "sources": sources}
        )

        st.session_state.historique_global.append({
            "user": st.session_state.username,
            "question": question,
            "date": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        })
# ============================================================
# DASHBOARD ADMIN
# ============================================================
def afficher_dashboard_admin():
    dossier_data = "data/raw"
    nb_docs = (
        len(os.listdir(dossier_data)) if os.path.exists(dossier_data) else 0
    )
    nb_questions_total = len(st.session_state.historique_global)
    nb_utilisateurs = len(USERS)

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{nb_docs}</div>'
            f'<div class="metric-label">Documents indexés</div></div>',
            unsafe_allow_html=True,
        )
    with col_m2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{nb_questions_total}</div>'
            f'<div class="metric-label">Questions (tous utilisateurs)</div></div>',
            unsafe_allow_html=True,
        )
    with col_m3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{nb_utilisateurs}</div>'
            f'<div class="metric-label">Comptes utilisateurs</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    onglet_docs, onglet_hist, onglet_comptes = st.tabs(
        ["📁 Gérer les documents", "📜 Historique global", "👥 Gérer les comptes"]
    )

    # ---- Onglet documents ----
    with onglet_docs:
        st.markdown(
            '<div class="section-title">Ajouter un document</div>',
            unsafe_allow_html=True,
        )
        fichier_televerse = st.file_uploader(
            "Ajouter un document",
            type=["pdf", "docx", "txt", "csv"],
            label_visibility="collapsed",
        )
        if fichier_televerse is not None:
            os.makedirs(dossier_data, exist_ok=True)
            chemin_dest = os.path.join(dossier_data, fichier_televerse.name)
            with open(chemin_dest, "wb") as f:
                f.write(fichier_televerse.getbuffer())
            st.success(f"Document « {fichier_televerse.name} » ajouté.")
            st.rerun()

        st.write("")
        st.markdown(
            '<div class="section-title">Documents existants</div>',
            unsafe_allow_html=True,
        )
        if os.path.exists(dossier_data):
            fichiers = sorted(os.listdir(dossier_data))
            if fichiers:
              for f in fichiers:
                    col_nom, col_suppr = st.columns([5, 1])
                    with col_nom:
                        with st.expander(f"📄 {f}"):
                            apercu = lire_apercu_document(os.path.join(dossier_data, f))
                            st.text(apercu + "...")
                    with col_suppr:
                        if st.button("🗑️", key=f"suppr_{f}"):
                            os.remove(os.path.join(dossier_data, f))
                            st.rerun()
            else:
                st.caption("Aucun document pour le moment.")
        else:
            st.caption("Aucun document trouvé.")

    # ---- Onglet historique global ----
    with onglet_hist:
        st.markdown(
            '<div class="section-title">Toutes les questions posées</div>',
            unsafe_allow_html=True,
        )
        if st.session_state.historique_global:
            for entree in reversed(st.session_state.historique_global):
                st.markdown(
                    f'<div class="hist-item"><b>{entree["user"]}</b> — {entree["question"]} '
                    f'<span style="color:{TEXT_MUTED}; font-size:11.5px;">({entree["date"]})</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Aucune question posée pour l'instant.")

    # ---- Onglet gestion des comptes ----
    with onglet_comptes:
        st.markdown(
            '<div class="section-title">Comptes existants</div>',
            unsafe_allow_html=True,
        )
        for u, infos in USERS.items():
            col_nom, col_role, col_suppr = st.columns([3, 2, 1])
            with col_nom:
                st.markdown(
                    f'<div class="doc-card">👤 {infos["nom"]} ({u})</div>',
                    unsafe_allow_html=True,
                )
            with col_role:
                st.markdown(
                    f'<span class="badge-role">{infos["role"]}</span>',
                    unsafe_allow_html=True,
                )
            with col_suppr:
                if u != "admin" and st.button("🗑️", key=f"suppr_user_{u}"):
                    del USERS[u]
                    st.rerun()

        st.write("")
        st.markdown(
            '<div class="section-title">Ajouter un compte</div>',
            unsafe_allow_html=True,
        )
        with st.form("form_ajout_compte"):
            c1, c2, c3 = st.columns(3)
            with c1:
                nouvel_identifiant = st.text_input("Identifiant")
            with c2:
                nouveau_mdp = st.text_input("Mot de passe")
            with c3:
                nouveau_role = st.selectbox("Rôle", ["user", "admin"])
            nouveau_nom = st.text_input("Nom complet")
            ajouter = st.form_submit_button("Ajouter le compte")

        if ajouter and nouvel_identifiant and nouveau_mdp:
            USERS[nouvel_identifiant] = {
                "password": nouveau_mdp,
                "role": nouveau_role,
                "nom": nouveau_nom or nouvel_identifiant,
            }
            st.success(f"Compte « {nouvel_identifiant} » créé.")
            st.rerun()


# ============================================================
# ROUTAGE PRINCIPAL
# ============================================================
if not st.session_state.authentifie:
    afficher_login()
else:
    afficher_topbar()
    if st.session_state.role == "admin":
        afficher_dashboard_admin()
    else:
        afficher_dashboard_utilisateur()