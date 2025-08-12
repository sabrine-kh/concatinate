# Workflow Détaillé - Transitions entre Étapes

## Workflow Principal avec Transitions

```mermaid
flowchart TD
    A[PDF Uploadé] --> B[Workflow Simple]
    B --> C[Étape 1: Web Scraping]
    C --> D[Recherche URLs]
    D --> E{URLs trouvées?}
    E -->|Oui| F[Scraping Contenu]
    E -->|Non| G[NOT FOUND - URLs]
    F --> H{Contenu extrait?}
    H -->|Oui| I[Succès - Web Scraping]
    H -->|Non| J[NOT FOUND - Contenu]
    G --> K[Transition vers NuExtract]
    J --> K
    K --> L[Étape 2: NuExtract]
    L --> M[Chargement Template]
    M --> N[Extraction avec NuMind]
    N --> O{Données extraites?}
    O -->|Oui| P[Succès - NuExtract]
    O -->|Non| Q[NOT FOUND - NuExtract]
    Q --> R[Transition vers LLM]
    R --> S[Étape 3: Recherche LLM]
    S --> T[Vector Store Search]
    T --> U[LLM Processing]
    U --> V{Contenu trouvé?}
    V -->|Oui| W[Succès - Recherche LLM]
    V -->|Non| X[Échec - Toutes étapes]
    
    I --> Y[Résultats]
    P --> Y
    W --> Y
    X --> Y
    
    style A fill:#e1f5fe
    style Y fill:#e8f5e8
    style X fill:#ffebee
    style I fill:#c8e6c9
    style P fill:#c8e6c9
    style W fill:#c8e6c9
```

## Détail des Conditions de Transition

```mermaid
flowchart TD
    A[Web Scraping] --> B{Condition de Sortie}
    B -->|"NOT FOUND"| C[Transition vers NuExtract]
    B -->|"ERROR"| C
    B -->|"TIMEOUT"| C
    B -->|"NO DATA"| C
    B -->|"SUCCESS"| D[Arrêt - Données Trouvées]
    
    C --> E[NuExtract]
    E --> F{Condition de Sortie}
    F -->|"NOT FOUND"| G[Transition vers LLM]
    F -->|"ERROR"| G
    F -->|"INVALID TEMPLATE"| G
    F -->|"NO MATCH"| G
    F -->|"SUCCESS"| H[Arrêt - Données Extraites]
    
    G --> I[Recherche LLM]
    I --> J{Condition de Sortie}
    J -->|"NOT FOUND"| K[Échec Final]
    J -->|"ERROR"| K
    J -->|"NO CONTEXT"| K
    J -->|"SUCCESS"| L[Arrêt - LLM Réussi]
    
    style D fill:#c8e6c9
    style H fill:#c8e6c9
    style L fill:#c8e6c9
    style K fill:#ffcdd2
```

## Logique de Décision Détaillée

```mermaid
flowchart TD
    A[Étape Actuelle] --> B[Exécution]
    B --> C{Statut Résultat}
    C -->|"SUCCESS"| D[Arrêter - Données Valides]
    C -->|"NOT FOUND"| E[Passer à l'étape suivante]
    C -->|"ERROR"| F[Gérer Erreur]
    C -->|"TIMEOUT"| G[Retry ou Passer]
    C -->|"NO DATA"| H[Passer à l'étape suivante]
    
    F --> I{Erreur Critique?}
    I -->|Oui| J[Passer à l'étape suivante]
    I -->|Non| K[Retry]
    K --> L{Max Retries?}
    L -->|Non| B
    L -->|Oui| E
    
    G --> M{Timeout Configuré?}
    M -->|Oui| E
    M -->|Non| K
    
    E --> N[Étape Suivante]
    N --> O{Étape Disponible?}
    O -->|Oui| P[Continuer Workflow]
    O -->|Non| Q[Échec Final]
    
    style D fill:#c8e6c9
    style Q fill:#ffcdd2
```

## Gestion des Erreurs par Étape

```mermaid
flowchart TD
    A[Web Scraping] --> B[Erreurs Possibles]
    B --> C[URLs non trouvées]
    B --> D[Site inaccessible]
    B --> E[Contenu protégé]
    B --> F[Timeout réseau]
    B --> G[Format non supporté]
    
    C --> H[NOT FOUND → NuExtract]
    D --> H
    E --> H
    F --> H
    G --> H
    
    I[NuExtract] --> J[Erreurs Possibles]
    J --> K[Template invalide]
    J --> L[API erreur]
    J --> M[Données non reconnues]
    J --> N[Format PDF corrompu]
    J --> O[Timeout API]
    
    K --> P[NOT FOUND → LLM]
    L --> P
    M --> P
    N --> P
    O --> P
    
    Q[Recherche LLM] --> R[Erreurs Possibles]
    R --> S[Vector store vide]
    R --> T[LLM non disponible]
    R --> U[Contexte insuffisant]
    R --> V[Prompt invalide]
    
    S --> W[ÉCHEC FINAL]
    T --> W
    U --> W
    V --> W
    
    style H fill:#fff3e0
    style P fill:#fff3e0
    style W fill:#ffebee
```

## Workflow avec Retry Logic

```mermaid
flowchart TD
    A[Étape Début] --> B[Premier Essai]
    B --> C{Succès?}
    C -->|Oui| D[Continuer]
    C -->|Non| E[Analyse Erreur]
    E --> F{Retry Possible?}
    F -->|Oui| G[Attendre + Retry]
    F -->|Non| H[Passer à l'étape suivante]
    
    G --> I{Max Retries?}
    I -->|Non| B
    I -->|Oui| H
    
    H --> J{Étape Suivante Existe?}
    J -->|Oui| K[Étape Suivante]
    J -->|Non| L[Échec Final]
    
    K --> M[Premier Essai]
    
    style D fill:#c8e6c9
    style L fill:#ffcdd2
``` 