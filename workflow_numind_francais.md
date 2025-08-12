# Workflow NuMind - Processus d'Extraction de Données

## Workflow Principal

```mermaid
flowchart TD
    A[Début] --> B[Initialisation NuMind]
    B --> C[Configuration API]
    C --> D[Chargement Template]
    D --> E[Validation Schéma]
    E --> F[Entrée PDF]
    F --> G[Prétraitement PDF]
    G --> H[PDF Traité]
    H --> I[Application Template]
    I --> J[Extraction Données]
    J --> K[Résultat: Données Extraites]
    K --> L[Fin]
    
    style A fill:#e1f5fe
    style L fill:#e8f5e8
    style K fill:#fff3e0
```

## Détail du Processus de Prétraitement

```mermaid
flowchart TD
    A[PDF Original] --> B[Conversion en Bytes]
    B --> C[Validation Format]
    C --> D{Format Valide?}
    D -->|Non| E[Erreur: Format Non Supporté]
    D -->|Oui| F[Nettoyage Contenu]
    F --> G[Extraction Texte]
    G --> H[Normalisation]
    H --> I[PDF Prétraité]
    I --> J[Prêt pour NuMind]
    
    style E fill:#ffebee
    style J fill:#e8f5e8
```

## Application du Template et Extraction

```mermaid
flowchart TD
    A[PDF Prétraité] --> B[Chargement Template]
    B --> C[Validation Schéma]
    C --> D[Application Règles]
    D --> E[Extraction Attributs]
    E --> F[Validation Données]
    F --> G[Formatage Résultats]
    G --> H[Données Extraites]
    
    style H fill:#fff3e0
```

## Schéma Template et Attributs

```mermaid
graph LR
    A[Template NuMind] --> B[Matériau]
    A --> C[Température]
    A --> D[Couleur]
    A --> E[Connecteurs]
    A --> F[Étanchéité]
    A --> G[Dimensions]
    
    B --> H[PA66, PBT, PA, etc.]
    C --> I[Min/Max °C]
    D --> J[Codes Couleur]
    E --> K[Types Contact]
    F --> L[Classes IP]
    G --> M[MM]
    
    style A fill:#e3f2fd
```

## Gestion des Erreurs

```mermaid
flowchart TD
    A[Processus Extraction] --> B{Succès?}
    B -->|Oui| C[Données Extraites]
    B -->|Non| D[Analyse Erreur]
    D --> E{Type Erreur?}
    E -->|API| F[Retry API]
    E -->|Template| G[Validation Template]
    E -->|PDF| H[Re-prétraitement]
    E -->|Données| I[Validation Schéma]
    
    F --> J{Max Retries?}
    J -->|Non| A
    J -->|Oui| K[Erreur Finale]
    
    style K fill:#ffebee
    style C fill:#e8f5e8
```

## Flux de Données Complet

```mermaid
graph TD
    A[Initialisation] --> B[Configuration]
    B --> C[Entrée PDF]
    C --> D[Prétraitement]
    D --> E[PDF Traité]
    E --> F[Template + Schéma]
    F --> G[Extraction NuMind]
    G --> H[Validation]
    H --> I[Données Extraites]
    
    subgraph "Configuration"
        B1[API Key]
        B2[Template]
        B3[Schéma]
    end
    
    subgraph "Prétraitement"
        D1[Conversion]
        D2[Nettoyage]
        D3[Normalisation]
    end
    
    subgraph "Extraction"
        G1[API Call]
        G2[Parsing]
        G3[Validation]
    end
    
    style I fill:#fff3e0
```

## Métriques et Performance

```mermaid
flowchart TD
    A[Début Traitement] --> B[Enregistrement Temps]
    B --> C[Initialisation]
    C --> D[Prétraitement PDF]
    D --> E[Extraction Données]
    E --> F[Calcul Performance]
    F --> G[Log Métriques]
    G --> H[Résultats + Stats]
    
    subgraph "Métriques"
        F1[Temps Total]
        F2[Succès/Échecs]
        F3[Qualité Extraction]
    end
    
    style H fill:#e8f5e8
```

## Intégration avec Système Existant

```mermaid
graph LR
    A[Web Scraping] --> B[Source Primaire]
    B --> C{Données Disponibles?}
    C -->|Oui| D[Utiliser Web]
    C -->|Non| E[NuMind Processing]
    E --> F[PDF Input]
    F --> G[Prétraitement]
    G --> H[Template + Schéma]
    H --> I[Extraction]
    I --> J[Source de Fallback]
    D --> K[Résultats Finaux]
    J --> K
    
    style K fill:#fff3e0
``` 