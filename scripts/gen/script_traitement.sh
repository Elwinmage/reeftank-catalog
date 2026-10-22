#!/bin/bash

# Fichiers par défaut modifiables
CSV_FILE="poissons_noms_multilingues.csv"
TXT_MODEL="modele.txt"
IDLE_MODEL="modele_idle.txt"
TURN_MODEL="modele_turn.txt"
SWIM_MODEL="modele_swim.txt"

# Surcharge possible via les arguments en ligne de commande
if [ $# -eq 5 ]; then
    CSV_FILE="$1"
    TXT_MODEL="$2"
    IDLE_MODEL="$3"
    TURN_MODEL="$4"
    SWIM_MODEL="$5"
fi

# Vérification de l'existence de tous les fichiers nécessaires
for file in "$CSV_FILE" "$TXT_MODEL" "$IDLE_MODEL" "$TURN_MODEL" "$SWIM_MODEL"; do
    if [ ! -f "$file" ]; then
        echo "Erreur : Le fichier '$file' est introuvable."
        exit 1
    fi
done

# Lecture du contenu des 4 modèles
TEMPLATE_CONTENT=$(<"$TXT_MODEL")
IDLE_CONTENT=$(<"$IDLE_MODEL")
TURN_CONTENT=$(<"$TURN_MODEL")
SWIM_CONTENT=$(<"$SWIM_MODEL")

# Compter le nombre total de lignes (hors en-tête) pour l'affichage indicatif
TOTAL_ROWS=$(($(wc -l < "$CSV_FILE") - 1))
echo "--- Début du traitement (Total brut : $TOTAL_ROWS lignes) ---"
echo ""

INDEX=0

# Fonction pour gérer la pause en lisant directement depuis le terminal (/dev/tty)
prompt_pause() {
    local message="$1"
    read -p "$message (ou tapez 'q' pour quitter) : " user_input </dev/tty
    if [[ "$user_input" =~ ^[Qq]$ ]]; then
        echo "Arrêt du script demandé par l'utilisateur."
        exit 0
    fi
    clear
}
clear
# Lecture du CSV ligne par ligne en ignorant l'en-tête
while IFS=',' read -r col1 rest; do
    # Nettoyage optionnel des guillemets
    col1="${col1//\"/}"

    # CONDITION : Si la ligne commence par un tiret '-', on l'ignore
    if [[ "$col1" == -* ]]; then
        echo -e "\033[33m[Ligne ignorée car commence par '-'] : $col1\033[0m"
        continue
    fi

    INDEX=$((INDEX + 1))

    echo -e "\033[32m========================================\033[0m"
    echo -e "\033[32m=== [Espèce $INDEX] : $col1 ===\033[0m"
    echo -e "\033[32m========================================\033[0m"

    # 1. Modèle principal
    echo "--- 1. Modèle principal ($TXT_MODEL) ---"
    echo "${TEMPLATE_CONTENT//\[SPECIE\]/$col1}"
    echo ""
    prompt_pause "Appuyez sur Entrée pour passer au modèle suivant"

    # 2. Modèle Idle
    echo "--- 2. Modèle Idle ($IDLE_MODEL) ---"
    echo "${IDLE_CONTENT//\[SPECIE\]/$col1}"
    echo ""
    prompt_pause "Appuyez sur Entrée pour passer au modèle suivant"

    # 3. Modèle Turn
    echo "--- 3. Modèle Turn ($TURN_MODEL) ---"
    echo "${TURN_CONTENT//\[SPECIE\]/$col1}"
    echo ""
    prompt_pause "Appuyez sur Entrée pour passer au modèle suivant"

    # 4. Modèle Swim
    echo "--- 4. Modèle Swim ($SWIM_MODEL) ---"
    echo "${SWIM_CONTENT//\[SPECIE\]/$col1}"
    echo ""

    # Pause de fin d'espèce
    prompt_pause "Fin de l'espèce [$col1]. Appuyez sur Entrée pour passer à l'espèce suivante"

done < <(tail -n +2 "$CSV_FILE")

echo "=== Fin du traitement de la liste CSV. ==="
