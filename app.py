diff --git a/app.py b/app.py
index 8c6bbe22d67b8e0365d184ecb7f53883e161a781..9c0a77f587e21b3ab08cc84fee3053614ed4c5f0 100644
--- a/app.py
+++ b/app.py
@@ -41,51 +41,51 @@ CONV_DIR = DATA_DIR / "conversations"
 STATIC_DIR = BASE_DIR / "static"
 
 CONV_DIR.mkdir(parents=True, exist_ok=True)
 STATIC_DIR.mkdir(parents=True, exist_ok=True)
 
 # Lecture des variables d’environnement
 def _env(key: str, default: str = "") -> str:
     return (os.getenv(key, default) or "").strip()
 
 # OpenAI
 OPENAI_API_KEY = _env("OPENAI_API_KEY")
 OPENAI_MODEL = _env("OPENAI_MODEL", "gpt-5.2")
 OPENAI_REASONING_EFFORT = _env("OPENAI_REASONING_EFFORT", "none")
 OPENAI_TEMPERATURE = float(_env("OPENAI_TEMPERATURE", "0.3") or 0.0)
 OPENAI_MAX_OUTPUT_TOKENS = int(_env("OPENAI_MAX_OUTPUT_TOKENS", "900"))
 
 # Sécurité admin / CORS
 ADMIN_API_TOKEN = _env("ADMIN_API_TOKEN")
 ADMIN_DASHBOARD_TOKEN = _env("ADMIN_DASHBOARD_TOKEN")
 ALLOWED_ORIGINS = [o for o in _env("ALLOWED_ORIGINS").split(",") if o.strip()]
 
 # Marque par défaut
 BRAND_ID_DEFAULT = _env("BRAND_ID", "retroworld").lower() or "retroworld"
 
 # FAQ publique : marques activées
-FAQ_ENABLED_BRANDS = [b for b in _env("FAQ_ENABLED_BRANDS", "retroworld,runningman").split(",") if b.strip()]
+FAQ_ENABLED_BRANDS = [b for b in _env("FAQ_ENABLED_BRANDS", "retroworld,runningman,enigmaniac").split(",") if b.strip()]
 # Marques proposées dans le widget public
 PUBLIC_BRANDS = [b for b in _env("PUBLIC_BRANDS", ",".join(FAQ_ENABLED_BRANDS)).split(",") if b.strip()]
 
 # URL publique (utilisée dans /brands.json)
 PUBLIC_BASE_URL = _env("PUBLIC_BASE_URL")
 
 # Logs de debug
 DEBUG_LOGS = _env("DEBUG_LOGS").lower() in ("1", "true", "yes", "on")
 
 # Informations par établissement (nom, contacts, domaines…)
 DEFAULT_BRANDS: Dict[str, Dict[str, Any]] = {
     "retroworld": {
         "name": "Retroworld",
         "short": "Retroworld",
         "contact_phone": "04 94 47 94 64",
         "contact_email": "contact@retroworldfrance.com",
         "website": "https://www.retroworldfrance.com",
         "domains": ["retroworldfrance.com", "www.retroworldfrance.com"],
     },
     "runningman": {
         "name": "Runningman",
         "short": "Runningman",
         "contact_phone": "04 98 09 30 59",
         "contact_email": "",
         "website": "https://www.runningmangames.fr",
diff --git a/src/data/system_data.py b/src/data/system_data.py
index 30018832a82d547a9abfbb88f5479136ac01b697..6a277ac357edfdb1b9311072684c716f6fd13734 100644
--- a/src/data/system_data.py
+++ b/src/data/system_data.py
@@ -1,119 +1,117 @@
-"""
-Prompt système utilisé par l’IA d’accueil du Pôle Loisirs Draguignan.
-Modifiez ce texte pour adapter les informations et règles commerciales à votre contexte.
-"""
+"""Prompt système officiel utilisé par l'IA d'accueil."""
 
-# Contenu du prompt :
 SYSTEM_PROMPT = """
-TU ES L'IA D'ACCUEIL EXPERTE DU "PÔLE LOISIRS DRAGUIGNAN".
-Tu gères 3 entités : Retroworld (VR), Runningman (Action/Enfant) et Enigmaniac (Escape Réel).
-
---- 🧠 RÈGLES DE VENTE & SÉCURITÉ ---
-1. **PRIX & RÉSERVATION** :
-   - Ne donne jamais de prix "environ". Sois précis.
-   - Pour Retroworld, donne les liens Qweekle.
-   - Pour Runningman/Enigmaniac, renvoie vers leur site ou téléphone.
-2. **MALADIE VR (NAUSÉE)** :
-   - Si client sensible : INTERDICTION des jeux à déplacement fluide (Walking Dead, Propagation Top Squad).
-   - CONSEIL : Jeux statiques UNIQUEMENT (Smash Point, Ragnarock, Pixel Hack, Archer).
-3. **PAS D'INVENTION** : Base-toi uniquement sur les descriptions ci-dessous.
-
---- 🔴 ENTITÉ 1 : RETROWORLD (VR & QUIZ) ---
-*Contact : 04 94 47 94 64 | Site : retroworldfrance.com*
+TU ES L'IA D'ACCUEIL UNIFIÉE DU "PÔLE LOISIRS DRAGUIGNAN".
+Tu gères l'accueil pour 3 entités situées au 815 av Pierre Brossolette (et Foch pour Enigmaniac).
+
+TES SOURCES DE VÉRITÉ OFFICIELLES (NE RIEN INVENTER) :
+- Retroworld (VR/Quiz) : retroworldfrance.com
+- Runningman (Action/Enfant) : runningmangames.fr
+- Enigmaniac (Escape Réel) : enigmaniac-escapegame.com
 
-=== FORMULES ANNIVERSAIRE ===
-- **Tarif** : Jeux VR (-5%), Escape VR (-5%), ou Combo VR + Quiz (20€/pers).
-- **Option Goûter** : 60€ (jusqu'à 10 pers, +5€/pers supp).
-  - Inclus : Crêpes, gaufres, glaces à volonté. (Résa 2 semaines avant).
-
-=== 1. JEUX VR ARCADE (15€/30min | +5€ hors créneaux) ===
-*Lien : https://retroworld.qweekle.com/shop/retroworld/multi/jeux-a-la-partie?tag=Jeu%20%C3%A0%20la%20partie&lang=fr*
-
-[POUR LA FAMILLE & DÉBUTANTS (Zéro Nausée)]
-- **Smash Point** : (Tir) Comme un Paintball cartoon super fun. Tout le monde adore. (1-5j).
-- **Pixel Hack** : (Tir) Défendez une zone contre des pixels. Style rétro arcade. (1-4j).
-- **Jolly Island** : (Aventure) Balade facile sur une île pirate. Idéal enfants. (1-5j).
-- **Clash of Chefs** : (Cuisine) Préparez des burgers ou pizzas en coop. Très drôle. (1-2j).
-- **Yin** : (Puzzle) Le jeu du "Serpent" (Snake) en 3D. Calme et joli. (2-4j).
-- **Angry Birds / Fruit Ninja** : Les classiques du mobile en géant. (Solo).
-
-[MUSIQUE & RYTHME (Zéro Nausée)]
-- **Ragnarock** : (Musique) Tapez sur des tambours vikings au rythme du métal/rock celtique. Course de drakkars. (1-5j).
-- **Rhythmatic 2** : (Danse) Tranchez des cubes en musique. Très physique ! (1-5j).
-
-[ACTION & TIR (Ados/Adultes)]
-- **Gang of Dummizz** : (Braquage) Braquez une banque avec des personnages maladroits. Fun. (1-5j).
-- **Archer** (Elven Assassin) : (Tir à l'arc) Défendez votre château contre des orcs. (1-5j).
-- **Arvi Arena / Revol VR3** : (Tir) Affrontez-vous dans une arène futuriste. (PVP 1-5j).
-- **Gunslinger** : (Western) Duel de cowboys. Rapide et nerveux. (1-5j).
-- **Battle Magic** : (Fantastique) Lancez des sorts pour battre vos amis. (2-5j).
-- **Head Gun 2** : (Action) Jeu de tir arcade déjanté où on guide avec la tête. (1-4j).
-- **Battle Wake** : (Pirate) Combat naval intense, vous incarnez un capitaine pirate magique. (Solo).
-
-[HORREUR & ZOMBIES (Âmes sensibles s'abstenir !)]
-- **Propagation (Stage 1, 2, 3)** : (Horreur) Vous êtes coincé dans un métro avec des zombies. Statique (pas de nausée) mais terrifiant. (1-5j).
-- **Propagation Top Squad** : (Action) Une escouade militaire nettoie une ville. Ça bouge (Attention nausée). (1-5j).
-- **Darkensum** : (Aventure Sombre) Combattez des monstres dans un monde parallèle. (1-5j).
-- **Rotten Apple** : (Zombie) Coopération pour survivre à l'apocalypse. (1-5j).
-- **The Walking Dead Onslaught** : (Survie) Le jeu officiel. Très violent et difficile. (Solo, Expert).
-
-=== 2. ESCAPE GAME VR (30€/1h | +5€ hors créneaux) ===
-*Lien : https://retroworld.qweekle.com/shop/retroworld/multi/jeux-a-la-partie?tag=escape%20game&lang=fr*
-
-[LES LICENCES UBISOFT (Graphismes AAA)]
-- **La Pyramide Perdue (Assassin's Creed)** : Expédition en Egypte. Décors sublimes. (2-4j).
-- **Au-delà de la porte de Méduse (Assassin's Creed)** : Grèce antique, bateau et grotte. (2-4j).
-- **Prince of Persia (Dagger of Time)** : Maîtrisez le temps pour résoudre les énigmes. (2-4j).
-- **Sauver Notre-Dame** : Incarnez des pompiers lors de l'incendie de la cathédrale. Historique. (2-4j).
-
-[AVENTURE & FANTASTIQUE (Tous publics)]
-- **Alice** : Le Pays des Merveilles. Magique, on grandit/rétrécit. (Difficile, 1-5j).
-- **Jungle Quest** : Style Jumanji. Animaux, îles volantes. (Facile, Idéal Famille, 1-5j).
-- **Christmas (Noël)** : Sauvez le Père Noël. Féerique pour les enfants. (Facile, 1-5j).
-- **Atlantis** : Explorez la cité engloutie sous l'eau. (2-4j).
-- **Dream Hackers (1, 2, 3)** : Entrez dans les rêves façon "Inception". Techno/Futuriste. (1-4j).
-- **Signal Lost** : Station spatiale à réparer. Ambiance Gravity. (Facile, 1-5j).
-- **Jumpers VR** : Aventure dynamique. (2-4j).
-
-[FRISSON & ENQUÊTE (Ados/Adultes)]
-- **House of Fear (1, 2, 3)** : Manoir hanté, ambiance film d'horreur. Très populaire. (1-4j).
-- **Sanctum** : Enquête sombre style Lovecraft. (1-5j).
-- **The Prison** : Évadez-vous d'une cellule avant l'exécution. (2-5j).
-- **Chernobyl** : Voyagez dans le temps à Pripyat. Historique et mystérieux. (1-5j).
-- **Lockdown VR (Temple, Circus, Kidnapped)** : 3 scénarios très difficiles pour experts. (1-5j).
-- **Call of Blood** / **Cursed Soul** : Enigmes et frissons.
-
-=== 3. QUIZ INTERACTIF (RETROWORLD) ===
-*Lien : https://retroworld.qweekle.com/shop/retroworld/multi/jeux-a-la-partie?tag=quizz&lang=fr*
-- **Concept** : Plateau TV, 9 manches (Blind test, Vrai/Faux, Rapidité). Joker et coups bas permis !
-- **Tarifs** : 8€ (30min) | 15€ (60min) | 20€ (90min). (+5€ hors créneaux).
-- **Joueurs** : 4 à 12 pupitres.
-
---- 🔵 ENTITÉ 2 : RUNNINGMAN (ACTION & ENFANT) ---
+--- 🚨 RÈGLE D'OR : SANTÉ & SÉCURITÉ VR ---
+Si le client mentionne "MALADE", "NAUSÉE", "PEUR DE VOMIR" :
+1. **INTERDIT** : Walking Dead, Propagation, Epic Roller Coaster, Jeux de course.
+2. **CONSEIL** : Propose UNIQUEMENT des jeux statiques (Smash Point, Pixel Hack, Ragnarock, Clash of Chef).
+
+--- 🎯 STRATÉGIE D'AIGUILLAGE ---
+1. **VR / Digital ?** -> Direction RETROWORLD.
+2. **Sport / Action Physique / Enfant ?** -> Direction RUNNINGMAN.
+3. **Escape Game ?** -> Demande toujours : "En Réalité Virtuelle (Retroworld) ou en Vrai avec décors réels (Enigmaniac) ?"
+
+--- 🔴 RETROWORLD : JEUX VR & QUIZ ---
+*Contact : 04 94 47 94 64 | Site : retroworldfrance.com*
+*Résa : Liens Qweekle ci-dessous uniquement.*
+
+=== 1. JEUX VR ARCADE (15€/30min | +5€ hors créneaux standard) ===
+*Lien Résa : https://retroworld.qweekle.com/shop/retroworld/multi/jeux-a-la-partie?tag=Jeu%20%C3%A0%20la%20partie&lang=fr*
+
+[TOP FAMILLE & DÉBUTANT (Zéro Nausée)]
+- **Smash Point** (Tir cartoon, Fun, 1-5j).
+- **Pixel Hack** (Action/Tir retro, 1-4j).
+- **Jolly Island** (Aventure enfant, Très facile, 1-5j).
+- **Angry Birds** / **Fruit Ninja** (Les classiques).
+- **Clash of Chef** (Cuisine, Fun, 1-2j).
+- **Yin** (Snake en VR, 2-4j).
+
+[MUSIQUE & RYTHME]
+- **Ragnarock** (Viking/Drum, Top vente, 1-5j).
+- **Rhythmatic 2** (Danse/Rythme, 1-5j).
+
+[ACTION / TIR (Pour joueurs à l'aise)]
+- **Gang of Dummizz** (Braquage, Fun, 1-5j).
+- **Arvi Arena** / **Revol VR3** / **Gunslinger** (Tir Arcade, 1-5j).
+- **Archer** (Tir à l'arc, Défense, 1-5j).
+- **Head Gun 2** (Nouveau ! Action fun).
+- **Battle Magic** (Magie, PvP, 2-5j).
+- **Battle Wake** (Pirate, Combat naval, Solo).
+
+[HORREUR / ZOMBIE (Âmes sensibles s'abstenir !)]
+- **Propagation** (Stage 1, 2, 3 + Top Squad + Top Survivor). LA référence horreur (1-5j).
+- **Darkensum** (Action/Horreur, 1-5j).
+- **Rotten Apple** (Coop Zombie, 1-5j).
+- **The Walking Dead Onslaught** (Solo, Expert, Survie).
+- **Last Day Defence** (Stratégie/Tir).
+
+=== 2. ESCAPE GAME VR (30€/1h | +5€ hors créneaux standard) ===
+*Lien Résa : https://retroworld.qweekle.com/shop/retroworld/multi/jeux-a-la-partie?tag=escape%20game&lang=fr*
+
+[LICENCES UBISOFT (Les Blockbusters)]
+- **Assassin's Creed** : "La Pyramide Perdue" & "Au-delà de la porte de Méduse".
+- **Prince of Persia** : "The Dagger of Time".
+- **Sauver Notre-Dame en Feu** (Pompier/Historique).
+
+[FANTASTIQUE / AVENTURE]
+- **Alice** (Pays des merveilles, Difficile).
+- **Jungle Quest** (Animaux, Facile, Familial).
+- **Atlantis** (Sous-marin, 2-4j).
+- **Dream Hackers** (Chapitres 1, 2 et 3 - Rêves/Hackers).
+- **Signal Lost** (SF/Espace, Facile).
+- **Star Force** (Espace, Action).
+- **Jumpers VR** (Nouveau !).
+- **Midori** (Aventure asiatique).
+
+[HORREUR / THRILLER]
+- **House of Fear** (Manoir hanté).
+- **Sanctum** (Enquête Lovecraftienne).
+- **Lockdown VR** (3 chapitres : Temple, Circus, Kidnapped - Difficile).
+- **Tchernobyl** (Historique/Radioactif).
+- **The Prison** (Évasion carcérale).
+- **Call of Blood** / **Cursed Soul** (Horreur).
+
+=== 3. QUIZ INTERACTIF ===
+- *Lien Résa : https://retroworld.qweekle.com/shop/retroworld/multi/jeux-a-la-partie?tag=quizz&lang=fr*
+- Buzzers, Ambiance TV, Blind tests.
+- 4 à 12 joueurs.
+- Tarifs : 8€ (30min) | 15€ (60min) | 20€ (90min).
+
+=== 4. ANNIVERSAIRES RETROWORLD ===
+- Formules dès 5 personnes.
+- Option Goûter (Crêpes/Gaufres à volonté) : 50€ (jusqu'à 10 pers).
+- Page info : retroworldfrance.com/notre-formules-anniversaire/
+
+--- 🔵 RUNNINGMAN : ACTION & ENFANTS ---
 *Contact : 04 98 09 30 59 | Site : runningmangames.fr*
 
 === SALLE ENFANT (KIDS ZONE) ===
-- **Tarif** : 50€/heure (Jusqu'à 10 enfants environ).
-- **Concept** : Espace privatif avec Mur interactif, Jeux en bois, Balayeuse.
-- **Âge** : 6 à 15 ans. Parfait pour anniversaires.
+- 50€/heure (demi-heure supp +25€).
+- Jeux en bois, mur interactif, balayeuse. Idéal 6-15 ans.
 
-=== ACTION GAME ===
-- **Concept** : Le "Sol est de lave" sur dalles lumineuses + Défis d'adresse.
-- **Infos** : Contacter pour devis/résa groupes.
+=== ACTION GAME (GAME ZONE) ===
+- Sol interactif, défis physiques (Le sol est de lave, Squid Game).
+- Tarifs : Packs groupes/anniversaire (voir site).
 
 === EXTRAS ===
 - Billard (10€/h), Baby-foot, Air Hockey.
 
---- 🟢 ENTITÉ 3 : ENIGMANIAC (ESCAPE RÉEL) ---
+--- 🟢 ENIGMANIAC : ESCAPE GAME RÉEL ---
 *Contact : 04 94 50 74 63 | Site : enigmaniac-escapegame.com*
-*Lieu : 5 Bd Foch & Pôle Loisirs Brossolette.*
+*Adresse : 5 Bd Maréchal Foch (Centre) & Pôle Loisirs Brossolette (Vérifier lors de la résa).*
 
-=== LES SCÉNARIOS (EN VRAI) ===
-1. **La Loi de la Jungle** (3-6j) : Vous êtes capturés par une tribu cannibale. Fuyez ! (Difficile 3/5).
-2. **Terreur Nocturne** (3-6j) : La poupée Becky hante les lieux. Angoissant. (Difficile 3/5).
+=== LES SALLES (DÉCORS RÉELS) ===
+1. **La Loi de la Jungle** (3-6j, Cannibales, Difficile 3/5).
+2. **Terreur Nocturne** (3-6j, Poupée Horreur, Difficile 3/5).
 
 === TARIFS ENIGMANIAC ===
-- 2 à 4 joueurs : 25€/pers.
-- 5 à 6 joueurs : 20€/pers.
-- Enfants (-12 ans) : 15€/pers.
-"""
\ No newline at end of file
+- 25€/pers (2-4j) | 20€/pers (5-6j) | 15€/pers (-12 ans).
+"""
diff --git a/static/chat-widget.html b/static/chat-widget.html
index 13f3a8c2b2c369c519f2b602563cc675b90b6dc6..c621d266486d36da409711c7d0971e39c7230dd0 100644
--- a/static/chat-widget.html
+++ b/static/chat-widget.html
@@ -1,370 +1,98 @@
 <!doctype html>
 <html lang="fr">
 <head>
   <meta charset="utf-8" />
   <meta name="viewport" content="width=device-width,initial-scale=1" />
-  <title>Retroworld IA</title>
+  <title>Pôle Loisirs IA</title>
   <style>
-    :root{
-      --bg:#0b0f14;
-      --card:#121824;
-      --border:rgba(255,255,255,.08);
-      --text:#e9eef5;
-      --muted:#9fb1ca;
-      --accent:#3b82f6;
-      --accent2:#f59e0b;
-      --radius:16px;
-      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
-      --sans: system-ui, -apple-system, Segoe UI, Roboto, Arial;
-    }
-    *{box-sizing:border-box}
-    body{
-      margin:0;
-      font-family:var(--sans);
-      background:
-        radial-gradient(900px 520px at 15% 0%, rgba(59,130,246,.18), transparent 55%),
-        radial-gradient(760px 520px at 90% 20%, rgba(245,158,11,.12), transparent 60%),
-        var(--bg);
-      color:var(--text);
-      height:100vh;
-      display:flex;
-      align-items:stretch;
-      justify-content:center;
-      padding:10px;
-    }
-    .shell{
-      width:100%;
-      max-width:760px;
-      background:rgba(18,24,36,.88);
-      border:1px solid rgba(255,255,255,.06);
-      border-radius:var(--radius);
-      overflow:hidden;
-      display:flex;
-      flex-direction:column;
-      box-shadow:0 12px 36px rgba(0,0,0,.4);
-    }
-    .top{
-      padding:12px 14px;
-      border-bottom:1px solid rgba(255,255,255,.06);
-      display:flex;
-      gap:10px;
-      align-items:center;
-      justify-content:space-between;
-      background:rgba(11,15,20,.55);
-      backdrop-filter:blur(10px);
-    }
-    .brand{
-      display:flex;
-      align-items:center;
-      gap:10px;
-      font-weight:900;
-      letter-spacing:.2px;
-    }
-    .dot{
-      width:12px;height:12px;border-radius:999px;
-      background:linear-gradient(135deg,#ff0066,#ffcc00);
-      box-shadow:0 8px 20px rgba(0,0,0,.4);
-    }
-    .sub{
-      color:var(--muted);
-      font-size:12px;
-      font-weight:600;
-    }
-    .controls{display:flex; gap:10px; align-items:center;}
-    select, button{
-      font:inherit;
-    }
-    select{
-      padding:9px 10px;
-      border-radius:12px;
-      background:#0b1220;
-      border:1px solid rgba(255,255,255,.12);
-      color:var(--text);
-      outline:none;
-      font-weight:700;
-    }
-    button{
-      padding:9px 10px;
-      border-radius:12px;
-      border:1px solid rgba(255,255,255,.12);
-      background:rgba(255,255,255,.04);
-      color:var(--text);
-      cursor:pointer;
-      font-weight:800;
-    }
-    button.primary{background:var(--accent); border-color:rgba(59,130,246,.25)}
-    button:disabled{opacity:.6; cursor:not-allowed}
-
-    .quick{
-      padding:10px 14px;
-      border-bottom:1px solid rgba(255,255,255,.06);
-      display:flex;
-      gap:8px;
-      flex-wrap:wrap;
-      background:rgba(13,19,31,.30);
-    }
-    .chip{
-      font-size:12px;
-      padding:7px 10px;
-      border-radius:999px;
-      border:1px solid rgba(255,255,255,.10);
-      background:rgba(255,255,255,.04);
-      color:var(--text);
-      cursor:pointer;
-      user-select:none;
-    }
-    .chip:hover{border-color:rgba(255,255,255,.22)}
-
-    .log{
-      padding:14px;
-      overflow:auto;
-      flex:1;
-      display:flex;
-      flex-direction:column;
-      gap:10px;
-    }
-    .bubble{
-      max-width:85%;
-      padding:10px 12px;
-      border-radius:14px;
-      border:1px solid rgba(255,255,255,.10);
-      background:rgba(17,24,39,.55);
-      white-space:pre-wrap;
-      line-height:1.35;
-      font-size:13px;
-    }
-    .bubble.user{margin-left:auto; background:rgba(30,41,59,.75)}
-    .bubble.bot{margin-right:auto; background:rgba(13,19,31,.55)}
-    .meta{
-      font-size:11px;
-      color:var(--muted);
-      font-family:var(--mono);
-      margin-top:2px;
-    }
-
-    .composer{
-      padding:12px 14px;
-      border-top:1px solid rgba(255,255,255,.06);
-      display:flex;
-      gap:10px;
-      background:rgba(11,15,20,.55);
-      backdrop-filter:blur(10px);
-    }
-    textarea{
-      flex:1;
-      resize:none;
-      height:44px;
-      min-height:44px;
-      max-height:140px;
-      padding:10px 12px;
-      border-radius:14px;
-      border:1px solid rgba(255,255,255,.12);
-      background:#0b1220;
-      color:var(--text);
-      outline:none;
-      line-height:1.3;
-    }
-    .toast{
-      position:fixed;
-      bottom:12px;
-      left:50%;
-      transform:translateX(-50%);
-      background:rgba(0,0,0,.75);
-      border:1px solid rgba(255,255,255,.12);
-      color:var(--text);
-      padding:10px 12px;
-      border-radius:999px;
-      font-size:12px;
-      opacity:0;
-      pointer-events:none;
-      transition:opacity .15s ease;
-    }
-    .toast.show{opacity:1}
+    :root{--bg:#0b0f14;--card:#121824;--border:rgba(255,255,255,.08);--text:#e9eef5;--muted:#9fb1ca;--accent:#3b82f6;--radius:16px;--sans:system-ui,-apple-system,Segoe UI,Roboto,Arial;}
+    *{box-sizing:border-box} body{margin:0;font-family:var(--sans);background:var(--bg);color:var(--text);height:100vh;display:flex;justify-content:center;padding:10px}
+    .shell{width:100%;max-width:780px;background:rgba(18,24,36,.92);border:1px solid rgba(255,255,255,.08);border-radius:var(--radius);display:flex;flex-direction:column;overflow:hidden}
+    .top,.composer,.tabs,.faq-wrap{padding:12px 14px;border-bottom:1px solid rgba(255,255,255,.08)} .composer{border-top:1px solid rgba(255,255,255,.08);border-bottom:0;display:flex;gap:8px}
+    .top{display:flex;justify-content:space-between;align-items:center}.controls{display:flex;gap:8px;align-items:center}
+    select,button,textarea{font:inherit} select,button{padding:8px 10px;border-radius:10px;background:#0b1220;border:1px solid rgba(255,255,255,.16);color:var(--text)}
+    button.primary{background:var(--accent)} button.tab{background:transparent} button.tab.active{background:#1e293b;border-color:#334155}
+    .tabs{display:flex;gap:8px;background:rgba(13,19,31,.35)}
+    .log{padding:14px;overflow:auto;flex:1;display:flex;flex-direction:column;gap:10px;min-height:240px}
+    .bubble{max-width:86%;padding:10px 12px;border-radius:14px;border:1px solid rgba(255,255,255,.10);white-space:pre-wrap;line-height:1.35;font-size:13px}
+    .bubble.user{margin-left:auto;background:rgba(30,41,59,.75)} .bubble.bot{margin-right:auto;background:rgba(13,19,31,.55)}
+    textarea{flex:1;resize:none;height:44px;padding:10px 12px;border-radius:12px;border:1px solid rgba(255,255,255,.16);background:#0b1220;color:var(--text)}
+    .faq-wrap{display:none;overflow:auto;flex:1}.faq-wrap.active{display:block}.faq-item{padding:10px;border:1px solid rgba(255,255,255,.10);border-radius:10px;margin-bottom:8px;background:rgba(13,19,31,.4)}
+    .faq-q{font-weight:700;margin-bottom:6px}.muted{color:var(--muted);font-size:12px}
   </style>
 </head>
 <body>
   <div class="shell">
     <div class="top">
-      <div>
-        <div class="brand"><span class="dot"></span> Retroworld IA</div>
-        <div class="sub">Réponses automatiques (VR, escape VR, quiz, anniversaires)</div>
-      </div>
+      <div><strong>Pôle Loisirs Draguignan IA</strong><div class="muted">Retroworld · Runningman · Enigmaniac</div></div>
       <div class="controls">
-        <select id="brandSel" title="Établissement">
-          <option value="auto">Auto</option>
-          <option value="retroworld">Retroworld</option>
-          <option value="runningman">Runningman</option>
-          <option value="enigmaniac">Enigmaniac (FAQ à venir)</option>
-        </select>
-        <button id="btnReset" title="Nouvelle conversation">Nouveau</button>
+        <select id="brandSel"><option value="auto">Auto</option><option value="retroworld">Retroworld</option><option value="runningman">Runningman</option><option value="enigmaniac">Enigmaniac</option></select>
+        <button id="btnReset">Nouveau</button>
       </div>
     </div>
 
-    <div class="quick">
-      <div class="chip" data-q="Quels sont vos horaires ?">Horaires</div>
-      <div class="chip" data-q="Quels sont vos tarifs ?">Tarifs</div>
-      <div class="chip" data-q="Où êtes-vous situés ?">Adresse</div>
-      <div class="chip" data-q="Je veux réserver pour un anniversaire, comment faire ?">Anniversaire</div>
-      <div class="chip" data-q="Combien de joueurs peuvent jouer en même temps ?">Joueurs</div>
-      <div class="chip" data-q="Comment se passe l'hygiène des casques VR ?">Hygiène</div>
+    <div class="tabs">
+      <button class="tab active" id="tabChat">Chat</button>
+      <button class="tab" id="tabFaq">FAQ</button>
     </div>
 
-    <div id="log" class="log"></div>
+    <div id="chatWrap" class="log"></div>
+    <div id="faqWrap" class="faq-wrap"></div>
 
-    <div class="composer">
-      <textarea id="input" placeholder="Écrivez votre message… (Entrée pour envoyer, Maj+Entrée pour une ligne)" ></textarea>
+    <div id="composer" class="composer">
+      <textarea id="input" placeholder="Écrivez votre message…"></textarea>
       <button class="primary" id="btnSend">Envoyer</button>
     </div>
   </div>
 
-  <div class="toast" id="toast"></div>
-
   <script>
     const $ = (id)=>document.getElementById(id);
+    const state={conversation_id:localStorage.getItem('rw_widget_conv_id')||'',sending:false};
 
-    const LS_CONV = "rw_widget_conv_id";
-    const LS_BRAND = "rw_widget_brand";
-
-    const state = {
-      conversation_id: localStorage.getItem(LS_CONV) || "",
-      sending:false
-    };
-
-    function toast(msg){
-      const t=$("toast");
-      t.textContent=msg;
-      t.classList.add("show");
-      setTimeout(()=>t.classList.remove("show"), 1400);
-    }
-
-    function addBubble(kind, text){
-      const log=$("log");
-      const b=document.createElement("div");
-      b.className="bubble " + (kind==="user"?"user":"bot");
-      b.textContent = text;
-      log.appendChild(b);
-      log.scrollTop = log.scrollHeight;
-    }
-
-    // Détermine l’URL d’envoi. Toutes les marques passent désormais par /chat ;
-    // le paramètre brand_id est envoyé dans le corps JSON.
-    function endpointFor(brand){
-      return "/chat";
-    }
+    function addBubble(kind,text){const b=document.createElement('div');b.className='bubble '+(kind==='user'?'user':'bot');b.textContent=text;$('chatWrap').appendChild(b);$('chatWrap').scrollTop=$('chatWrap').scrollHeight;}
 
     async function sendMessage(text){
-      const brand=$("brandSel").value;
-      if(!text || !text.trim()) return;
-      if(state.sending) return;
-      state.sending=true;
-      $("btnSend").disabled=true;
-
-      addBubble("user", text.trim());
-
-      try{
-        const res = await fetch(endpointFor(brand), {
-          method:"POST",
-          headers:{"Content-Type":"application/json"},
-          body: JSON.stringify({
-            message: text.trim(),
-            conversation_id: state.conversation_id || undefined,
-            // utile si le widget est embarqué avec ?brand=xxx
-            brand_id: (brand && brand !== "auto") ? brand : undefined
-          })
-        });
-        const data = await res.json().catch(()=>null);
-        if(!res.ok){
-          throw new Error((data && (data.error||data.message)) || "Erreur serveur");
-        }
-        if(data && data.conversation_id){
-          state.conversation_id = data.conversation_id;
-          localStorage.setItem(LS_CONV, state.conversation_id);
-        }
-        // Le serveur renvoie la réponse dans la propriété `answer`.
-        addBubble("bot", (data && data.answer) ? data.answer : "(Réponse vide)");
-      }catch(e){
-        addBubble("bot", "Désolé, je n’arrive pas à répondre pour le moment.\n" + (e.message || e));
-      }finally{
-        state.sending=false;
-        $("btnSend").disabled=false;
-      }
-    }
-
-    // quick chips
-    document.querySelectorAll('.chip').forEach(el=>{
-      el.addEventListener('click', ()=>{
-        const q = el.getAttribute('data-q') || "";
-        $("input").value = q;
-        $("input").focus();
-      });
-    });
-
-    // send
-    $("btnSend").addEventListener('click', ()=>{
-      const v=$("input").value;
-      $("input").value="";
-      sendMessage(v);
-    });
-
-    $("input").addEventListener('keydown', (ev)=>{
-      if(ev.key === 'Enter' && !ev.shiftKey){
-        ev.preventDefault();
-        const v=$("input").value;
-        $("input").value="";
-        sendMessage(v);
-      }
-    });
-
-    // reset
-    $("btnReset").addEventListener('click', ()=>{
-      state.conversation_id="";
-      localStorage.removeItem(LS_CONV);
-      $("log").innerHTML="";
-      toast("Nouvelle conversation");
-    });
-
-    // brand from URL (?brand=retroworld|runningman|enigmaniac)
-    const urlBrand = new URLSearchParams(location.search).get('brand') || new URLSearchParams(location.search).get('brand_id');
-    if(urlBrand){
-      $("brandSel").value = urlBrand;
-      localStorage.setItem(LS_BRAND, urlBrand);
-    }
-
-    // persist brand
-    const savedBrand = localStorage.getItem(LS_BRAND);
-    if(savedBrand && !urlBrand){ $("brandSel").value = savedBrand; }
-    $("brandSel").addEventListener('change', ()=>{
-      localStorage.setItem(LS_BRAND, $("brandSel").value);
-      toast("Mode: " + $("brandSel").value);
-    });
-
-    // try to auto-populate brand list from server (/brands.json)
-    (async ()=>{
+      const brand=$('brandSel').value;if(!text.trim()||state.sending)return;state.sending=true;$('btnSend').disabled=true;addBubble('user',text.trim());
       try{
-        const res = await fetch('/brands.json');
-        const data = await res.json();
-        if(!data || !Array.isArray(data.brands)) return;
-        const sel = $("brandSel");
-        const current = sel.value;
-        const keep = new Set(["auto"]);
-        data.brands.forEach(b=>keep.add(String(b.id||"")));
-        // rebuild options
-        sel.innerHTML = '';
-        const optAuto = document.createElement('option');
-        optAuto.value='auto'; optAuto.textContent='Auto';
-        sel.appendChild(optAuto);
-        data.brands.forEach(b=>{
-          const o=document.createElement('option');
-          o.value=String(b.id||'');
-          o.textContent=String(b.display_name||b.name||b.id||'');
-          sel.appendChild(o);
-        });
-        sel.value = current;
-      }catch(e){ /* ignore */ }
+        const res=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:text.trim(),conversation_id:state.conversation_id||undefined,brand_id:brand!=='auto'?brand:undefined})});
+        const data=await res.json();if(!res.ok)throw new Error(data.error||'Erreur serveur');
+        if(data.conversation_id){state.conversation_id=data.conversation_id;localStorage.setItem('rw_widget_conv_id',state.conversation_id);} addBubble('bot',data.answer||'(Réponse vide)');
+      }catch(e){addBubble('bot','Erreur: '+(e.message||e));}
+      state.sending=false;$('btnSend').disabled=false;
+    }
+
+    async function loadFaq(){
+      const bid=$('brandSel').value==='auto'?'retroworld':$('brandSel').value;
+      const res=await fetch('/faq.json?brand_id='+encodeURIComponent(bid));
+      const data=await res.json();
+      const wrap=$('faqWrap');
+      wrap.innerHTML='';
+      const items=Array.isArray(data.items)?data.items:[];
+      if(!items.length){wrap.innerHTML='<div class="muted">Aucune FAQ disponible pour cette marque.</div>';return;}
+      items.forEach(it=>{const el=document.createElement('div');el.className='faq-item';el.innerHTML=`<div class="faq-q">${it.question||''}</div><div>${it.answer||''}</div>`;wrap.appendChild(el)});
+    }
+
+    function setTab(tab){
+      const chat=tab==='chat';$('tabChat').classList.toggle('active',chat);$('tabFaq').classList.toggle('active',!chat);
+      $('chatWrap').style.display=chat?'flex':'none';$('composer').style.display=chat?'flex':'none';$('faqWrap').classList.toggle('active',!chat);
+      if(!chat) loadFaq().catch(()=>{$('faqWrap').innerHTML='<div class="muted">Impossible de charger la FAQ.</div>';});
+    }
+
+    $('btnSend').onclick=()=>{const v=$('input').value;$('input').value='';sendMessage(v)};
+    $('input').addEventListener('keydown',(ev)=>{if(ev.key==='Enter'&&!ev.shiftKey){ev.preventDefault();const v=$('input').value;$('input').value='';sendMessage(v);}});
+    $('btnReset').onclick=()=>{state.conversation_id='';localStorage.removeItem('rw_widget_conv_id');$('chatWrap').innerHTML='';};
+    $('tabChat').onclick=()=>setTab('chat');$('tabFaq').onclick=()=>setTab('faq');
+
+    $('brandSel').addEventListener('change',()=>{localStorage.setItem('rw_widget_brand',$('brandSel').value); if($('faqWrap').classList.contains('active'))loadFaq();});
+    const savedBrand=localStorage.getItem('rw_widget_brand'); if(savedBrand)$('brandSel').value=savedBrand;
+    const urlBrand=new URLSearchParams(location.search).get('brand')||new URLSearchParams(location.search).get('brand_id'); if(urlBrand)$('brandSel').value=urlBrand;
+    const tabParam=(new URLSearchParams(location.search).get('tab')||'chat').toLowerCase(); setTab(tabParam==='faq'?'faq':'chat');
+
+    (async()=>{
+      try{const res=await fetch('/brands.json');const data=await res.json();const items=Array.isArray(data.items)?data.items:[];if(!items.length)return;const cur=$('brandSel').value;$('brandSel').innerHTML='<option value="auto">Auto</option>';items.forEach(b=>{const o=document.createElement('option');o.value=b.id;o.textContent=b.short||b.name||b.id;$('brandSel').appendChild(o);});$('brandSel').value=cur;}catch(e){}
     })();
 
-    // welcome
-    addBubble("bot", "Bonjour. Je peux répondre aux questions (horaires, tarifs, réservations, anniversaires) et vous orienter.\n\nIndiquez l’établissement si besoin (Retroworld / Runningman / Enigmaniac) et votre date si c’est une réservation.");
+    addBubble('bot','Bonjour 👋 Je peux vous orienter entre Retroworld, Runningman et Enigmaniac, et répondre avec les informations officielles du Pôle Loisirs.');
   </script>
 </body>
 </html>
diff --git a/static/faq_enigmaniac.json b/static/faq_enigmaniac.json
new file mode 100644
index 0000000000000000000000000000000000000000..b1cdb8a93eecfe05d5c2bc57eb41cb8694e29023
--- /dev/null
+++ b/static/faq_enigmaniac.json
@@ -0,0 +1,16 @@
+{
+  "brand": "enigmaniac",
+  "updated": "2026-01-01 00:00:00",
+  "items": [
+    {
+      "question": "Quels sont les tarifs Enigmaniac ?",
+      "answer": "25€/pers (2 à 4 joueurs), 20€/pers (5 à 6 joueurs), 15€/pers pour les moins de 12 ans.",
+      "tags": ["tarifs", "enigmaniac"]
+    },
+    {
+      "question": "Quelles salles sont disponibles ?",
+      "answer": "La Loi de la Jungle et Terreur Nocturne, toutes les deux en décors réels pour 3 à 6 joueurs.",
+      "tags": ["salles", "escape"]
+    }
+  ]
+}
