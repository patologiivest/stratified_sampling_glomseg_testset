# ============================================================================
# IMPORTS AND INITIALIZATION
# ============================================================================

import pandas as pd
from itables import init_notebook_mode

init_notebook_mode(all_interactive=True)


# ============================================================================
# CONSTANTS: STAIN MAPPINGS
# ============================================================================

IMMUNE_STAINS = {"Kappa", "Lambda", "IgM", "IgG", "IgA", "C3", "C5-9", "C1q"}
AFOG_CONGO_STAINS = {"AFOG", "Masson Trichrom"}
HE_HES_STAINS = {"HE", "HES"}
VALID_STAINS = {
    "PAS",
    "PASM",
    "HE/HES",
    "AFOG/Masson Trichrom",
    "Congo",
    "Toluidine blue",
    "Sirius red",
    "Immune",
}


# ============================================================================
# CONSTANTS: DIAGNOSIS MAPPINGS
# ============================================================================

MEMBRANOUS_GLOMERULONEPHRITIS_DIAGNOSES = [
    "Membranøs glomerulonefritt - ikke klassifisert",
    "Membranøs glomerulonefritt - idiopatisk",
    "Membranøs glomerulonefritt - assosiert med malign sykdom",
    "Membranøs glomerulonefritt - medikament assosiert",
    "Membranøs glomerulonefritt - assosiert med infeksjon",
]

AMYLOIDOSIS_DIAGNOSES = [
    "Amyloidose - ikke klassifisert",
    "Amyloidose - AA",
    "Amyloidose - AL",
    "Amyloidose - andere",
]

VALID_DIAGNOSES = {
    "Minimal change nefropati",
    "Fokal og segmental glomerulosklerose primær",
    "IgA nefropati",
    "Lupus nefritt - II mesangioproliferativ lupusnefritt",
    "Lupus nefritt - III fokal lupusnefritt",
    "Lupus nefritt - IV diffus lupusnefritt",
    "Lupus nefritt - V membranøs lupusnefritt",
    "Endokapillær glomerulonefritt",
    "Membranøs glomerulonefritt",
    "Membranoproliferativ glomerulonefritt",
    "Anti-GBM nefritt",
    "ANCA assosiert glomerulonefritt",
    "fokal GN med nekroser / halvmåner, halvmåneglomerulonefritt uten holdepunkter for annen grunnsykdom",
    "Henoch Schönlein's purpura",
    "Trombotisk mikroangiopati",
    "Malign nefrosklerose",
    "Benign nefrosklerose",
    "Diabetisk nefropati",
    "Amyloidose",
    "Hereditær nefropati - Fabry's sykdom",
    "Hereditær nefropati - Alport sykdom",
    "Hereditær nefropati - Tynn basalmembran sykdom",
    "Akutt tubulær nekrose - ikke klassifisert",
    "Tubulointerstitiell nefritt",
    "End stage kidney - skrumpnyre",
    "Annen glomerulonefritt\\nyresykdom - uklassifiserbar",
    "Normal eller svært lette og uspesifikke forandringer",
    "Ukarakteristiske atrofiforandringer",
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def prep_df_wsi(csv_path: str) -> pd.DataFrame:
    """
    Load and prepare WSI (Whole Slide Image) data from CSV.

    Args:
        csv_path: Path to WSI CSV file

    Returns:
        DataFrame with standardized column names and date formatting
    """
    wsi_df = pd.read_csv(csv_path)
    selected_columns = [
        "patient_fnr",
        "ANON_name",
        "Stain",
        "Captured Date",
        "biop_id",
        "slide_id",
    ]
    wsi_df = wsi_df[selected_columns]
    wsi_df = wsi_df.rename(
        columns={
            "ANON_name": "wsi_anon_name",
            "Stain": "stain",
            "Captured Date": "captured_date",
            "biop_id": "biopsy_id_wsi_record",
        }
    )
    wsi_df["captured_date"] = pd.to_datetime(wsi_df["captured_date"], errors="coerce")
    wsi_df["year"] = wsi_df["captured_date"].dt.strftime("%Y")
    return wsi_df


def map_stain_to_super_stain(stain: str) -> str:
    """
    Map individual stain values to coarser stain categories.

    Args:
        stain: Individual stain type

    Returns:
        Mapped super stain category
    """
    if stain in IMMUNE_STAINS:
        return "immune"
    elif stain in AFOG_CONGO_STAINS:
        return "AFOG/Masson Trichrom"
    elif stain in HE_HES_STAINS:
        return "HE/HES"
    else:
        return stain


def map_diagnosis(diagnosis: str) -> str:
    """
    Map diagnosis variants to standardized diagnosis categories.

    Args:
        diagnosis: Original diagnosis string

    Returns:
        Mapped diagnosis string
    """
    if diagnosis in MEMBRANOUS_GLOMERULONEPHRITIS_DIAGNOSES:
        return "Membranøs glomerulonefritt"
    elif diagnosis in AMYLOIDOSIS_DIAGNOSES:
        return "Amyloidose"
    else:
        return diagnosis


# ============================================================================
# DATA LOADING
# ============================================================================

# Load and prepare diagnosis data
all_patient_data_df = pd.read_csv("../csv_data/all_patient_data.csv")
selected_columns = [
    "PersNummer",
    "Biopsidato",
    "Diagnoser P1.2013_konklusiv_diagnose",
    "BiopsiID / Ny-K-tabell.Lopenr",
    "Lab_navn",
    "Ant_G/Antall glom",
    "Ant_G_HM/GlomHM",
]

diagnosis_df = all_patient_data_df[selected_columns]
diagnosis_df = diagnosis_df.rename(
    columns={
        "PersNummer": "patient_fnr",
        "Biopsidato": "biopsy_date",
        "Lab_navn": "lab_name",
        "Diagnoser P1.2013_konklusiv_diagnose": "diagnosis",
        "BiopsiID / Ny-K-tabell.Lopenr": "biopsy_id_patient_record",
        "Ant_G/Antall glom": "number_glom",
        "Ant_G_HM/GlomHM": "number_glom_crescent",
    }
)

# Load and prepare WSI data
wsi_KB_df = prep_df_wsi("../csv_data/KB_csv.csv")
wsi_NKBR_df = prep_df_wsi("../csv_data/NKBR_csv.csv")
merged_wsi_df = pd.concat([wsi_KB_df, wsi_NKBR_df], ignore_index=True)

# Load and prepare scanner data
scanner_df = pd.read_csv("../csv_data/scanner_manufacturer_info.csv")
scanner_df_left = pd.read_csv("../csv_data/scanner_manufacturer_info_left.csv")
scanner_df = scanner_df[["patho_slide_id", "patho_scanner_manufacturer"]].dropna()
scanner_df_left = scanner_df_left[
    ["patho_slide_id", "patho_scanner_manufacturer"]
].dropna()
scanner_df = pd.concat([scanner_df, scanner_df_left], ignore_index=True)


# ============================================================================
# DATA PROCESSING: DATES AND TYPES
# ============================================================================

# Process diagnosis dates
diagnosis_df["biopsy_date"] = pd.to_datetime(
    diagnosis_df["biopsy_date"], errors="coerce"
)
diagnosis_df["biopsy_date_str"] = diagnosis_df["biopsy_date"].dt.strftime("%Y")


# ============================================================================
# DATA MERGING
# ============================================================================

# Merge WSI with scanner information
merged_wsi_df = merged_wsi_df.merge(
    scanner_df, left_on="slide_id", right_on="patho_slide_id", how="left"
)

# Merge with diagnosis data
stratification_df = merged_wsi_df.merge(diagnosis_df, on=["patient_fnr"], how="left")


# ============================================================================
# DATA FILTERING AND CLEANING: LAB INFORMATION
# ============================================================================

stratification_df["lab_name"] = stratification_df["lab_name"].dropna()
stratification_df = stratification_df[stratification_df["lab_name"] != "ULLEVÅL"]
stratification_df["lab_name"] = stratification_df["lab_name"].replace(
    "ÅLESUN", "ÅLESUND"
)
stratification_df["lab_name"] = stratification_df["lab_name"].astype("category")


# ============================================================================
# DATA FILTERING AND CLEANING: STAIN INFORMATION
# ============================================================================

# Fix ST.OLAVS Sirius red to Congo
stratification_df = stratification_df.apply(
    lambda row: (
        row.copy()
        if not ((row["stain"] == "Sirius red") and (row["lab_name"] == "ST.OLAVS"))
        else row.update({"stain": "Congo"})
    ),
    axis=1,
)

# Map stains to super_stain categories
stratification_df["super_stain"] = stratification_df["stain"].apply(
    map_stain_to_super_stain
)

# Filter to valid stains
stratification_df = stratification_df[
    stratification_df["super_stain"].isin(VALID_STAINS | {"immune"})
]
stratification_df["super_stain"] = stratification_df["super_stain"].astype("category")


# ============================================================================
# DATA FILTERING AND CLEANING: DIAGNOSIS INFORMATION
# ============================================================================

stratification_df["diagnosis"] = stratification_df["diagnosis"].dropna()

# Map diagnosis variants to standardized categories
stratification_df["diagnosis"] = stratification_df["diagnosis"].apply(map_diagnosis)

# Filter to valid diagnoses
stratification_df = stratification_df[
    stratification_df["diagnosis"].isin(VALID_DIAGNOSES)
]
stratification_df["diagnosis"] = stratification_df["diagnosis"].astype("category")


# ============================================================================
# FINAL DATA CLEANING AND PREPARATION
# ============================================================================

# Filter by scanner manufacturer
stratification_df.dropna(subset=["patho_scanner_manufacturer"], inplace=True)

# Convert year and lab_name to categories
stratification_df["year"] = stratification_df["year"].astype("category")
stratification_df["lab_name"] = stratification_df["lab_name"].astype("category")

# Ensure number of crescent glomeruli <= number of glomeruli
stratification_df = stratification_df[
    stratification_df["number_glom"] >= stratification_df["number_glom_crescent"]
]

# Convert glomuleri counts to integer
stratification_df["number_glom_crescent"] = (
    stratification_df["number_glom_crescent"].fillna(0).astype(int)
)
stratification_df["number_glom"] = (
    stratification_df["number_glom"].fillna(0).astype(int)
)

# Create final output dataframe by dropping intermediate columns
df = stratification_df.drop(
    columns=[
        "captured_date",
        "biopsy_id_wsi_record",
        "biopsy_id_patient_record",
        "biopsy_date",
        "biopsy_date_str",
    ]
)

df.to_csv("../csv_data/stratification_data.csv", index=False)

diagnosis_weights = (
    stratification_df["diagnosis"].value_counts(normalize=True).to_dict()
)
