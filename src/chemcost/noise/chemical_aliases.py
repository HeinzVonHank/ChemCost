"""Chemical name aliases, abbreviations, and ambiguous isomer mappings.

Each mapping is designed to introduce realistic ambiguity that a working chemist
would encounter: abbreviated names, common/IUPAC switching, dropped prefixes,
and isomer-ambiguous parent names.
"""

# ── Isomer ambiguity ─────────────────────────────────────────────────────────
# Maps specific isomer names -> ambiguous parent name.  The parent name is
# chemically valid but refers to multiple isomers with different prices/MW.

ISOMER_AMBIGUOUS: dict[str, str] = {
    # Butanol isomers
    "n-butanol": "butanol",
    "n-BuOH": "butanol",
    "sec-butanol": "butanol",
    "tert-butanol": "butanol",
    "isobutanol": "butanol",
    "2-butanol": "butanol",
    "1-butanol": "butanol",
    # Propanol isomers
    "n-propanol": "propanol",
    "1-propanol": "propanol",
    "2-propanol": "propanol",
    "isopropanol": "propanol",
    "i-PrOH": "propanol",
    # Xylene isomers
    "o-xylene": "xylene",
    "m-xylene": "xylene",
    "p-xylene": "xylene",
    # Cresol isomers
    "o-cresol": "cresol",
    "m-cresol": "cresol",
    "p-cresol": "cresol",
    # Toluidine isomers
    "o-toluidine": "toluidine",
    "m-toluidine": "toluidine",
    "p-toluidine": "toluidine",
    # Dichlorobenzene isomers
    "1,2-dichlorobenzene": "dichlorobenzene",
    "1,3-dichlorobenzene": "dichlorobenzene",
    "1,4-dichlorobenzene": "dichlorobenzene",
    "o-dichlorobenzene": "dichlorobenzene",
    "m-dichlorobenzene": "dichlorobenzene",
    "p-dichlorobenzene": "dichlorobenzene",
    # Chloroaniline isomers
    "2-chloroaniline": "chloroaniline",
    "3-chloroaniline": "chloroaniline",
    "4-chloroaniline": "chloroaniline",
    # Nitrophenol isomers
    "2-nitrophenol": "nitrophenol",
    "3-nitrophenol": "nitrophenol",
    "4-nitrophenol": "nitrophenol",
    # Pentane isomers
    "n-pentane": "pentane",
    "isopentane": "pentane",
    "neopentane": "pentane",
    # Hexane isomers
    "n-hexane": "hexane",
    # Diethyl ether ambiguity
    "diethyl ether": "ether",
    # Lutidine isomers
    "2,6-lutidine": "lutidine",
    "2,4-lutidine": "lutidine",
    "3,5-lutidine": "lutidine",
    "2,6-Lutidine": "lutidine",
    # Picolines
    "2-picoline": "picoline",
    "3-picoline": "picoline",
    "4-picoline": "picoline",
    # Collidine
    "2,4,6-collidine": "collidine",
    "collidine": "trimethylpyridine",
}


# ── Stereochemistry prefixes to strip ────────────────────────────────────────
# Regex patterns for stereochemistry indicators that can be removed to
# create ambiguity.
STEREO_PREFIXES = [
    r"\(R\)-",
    r"\(S\)-",
    r"\(R,R\)-",
    r"\(S,S\)-",
    r"\(R,S\)-",
    r"\(S,R\)-",
    r"\(E\)-",
    r"\(Z\)-",
    r"\(±\)-",
    r"rac-",
    r"meso-",
    r"D-",
    r"L-",
    r"d-",
    r"l-",
    r"cis-",
    r"trans-",
    r"\(1R\)-",
    r"\(1S\)-",
    r"\(1R,2S\)-",
    r"\(1S,2R\)-",
    r"\(1R,2R\)-",
    r"\(1S,2S\)-",
    r"alpha-",
    r"beta-",
    r"α-",
    r"β-",
]

# Prefixes that indicate a specific isomer (non-regex, simple startswith)
POSITIONAL_PREFIXES = [
    "n-",
    "sec-",
    "tert-",
    "iso-",
    "neo-",
    "cyclo-",
    "o-",
    "m-",
    "p-",
]


# ── Abbreviation / common name pairs ─────────────────────────────────────────
# Bidirectional: full name <-> abbreviation.  Each entry is:
#   (specific_form, ambiguous_or_alternative_form)
# Direction of replacement depends on noise type: name_variation may go
# either way to test whether the agent can resolve both forms.

ABBREVIATION_TO_FULL: dict[str, list[str]] = {
    "DMF": ["dimethylformamide", "dimethyl formamide", "N,N-dimethylformamide"],
    "DMSO": ["dimethyl sulfoxide", "dimethylsulfoxide"],
    "DCM": ["dichloromethane", "methylene chloride"],
    "THF": ["tetrahydrofuran"],
    "MeCN": ["acetonitrile"],
    "MeOH": ["methanol"],
    "EtOH": ["ethanol"],
    "EtOAc": ["ethyl acetate"],
    "Et3N": ["triethylamine"],
    "TEA": ["triethylamine"],  # ambiguous: could also be triethanolamine
    "DIPEA": ["diisopropylethylamine", "N,N-diisopropylethylamine", "Hunig's base"],
    "AcOH": ["acetic acid"],
    "NBS": ["N-bromosuccinimide"],
    "NCS": ["N-chlorosuccinimide"],
    "DMAP": ["4-dimethylaminopyridine", "4-(dimethylamino)pyridine"],
    "DCC": ["dicyclohexylcarbodiimide", "N,N'-dicyclohexylcarbodiimide"],
    "EDC": ["1-ethyl-3-(3-dimethylaminopropyl)carbodiimide"],
    "HATU": [
        "hexafluorophosphate azabenzotriazole tetramethyl uronium",
        "1-[bis(dimethylamino)methylene]-1H-1,2,3-triazolo[4,5-b]pyridinium "
        "3-oxide hexafluorophosphate",
    ],
    "HOBt": ["1-hydroxybenzotriazole", "hydroxybenzotriazole"],
    "HOBT": ["1-hydroxybenzotriazole", "hydroxybenzotriazole"],
    "AIBN": ["azobisisobutyronitrile", "2,2'-azobis(2-methylpropionitrile)"],
    "DBU": ["1,8-diazabicyclo[5.4.0]undec-7-ene"],
    "mCPBA": ["meta-chloroperoxybenzoic acid", "3-chloroperoxybenzoic acid"],
    "m-CPBA": ["meta-chloroperoxybenzoic acid", "3-chloroperoxybenzoic acid"],
    "BINAP": ["2,2'-bis(diphenylphosphino)-1,1'-binaphthyl"],
    "PPh3": ["triphenylphosphine"],
    "n-BuLi": ["n-butyllithium", "butyllithium"],
    "LDA": ["lithium diisopropylamide"],
    "LAH": ["lithium aluminium hydride", "lithium aluminum hydride", "LiAlH4"],
    "NMO": ["N-methylmorpholine N-oxide", "4-methylmorpholine N-oxide"],
    "NMM": ["N-methylmorpholine", "4-methylmorpholine"],
    "TPAP": [
        "tetrapropylammonium perruthenate",
        "tetra-n-propylammonium perruthenate",
    ],
    "DIC": ["N,N'-diisopropylcarbodiimide", "diisopropylcarbodiimide"],
    "PhH": ["benzene"],
    "PCy3": ["tricyclohexylphosphine"],
    "XPhos": ["2-dicyclohexylphosphino-2',4',6'-triisopropylbiphenyl"],
    "t-BuOK": ["potassium tert-butoxide"],
    "HMPA": ["hexamethylphosphoramide", "hexamethylphosphoric triamide"],
}

# Reverse mapping: full name -> abbreviation(s)
FULL_TO_ABBREVIATION: dict[str, list[str]] = {}
for abbrev, fulls in ABBREVIATION_TO_FULL.items():
    for full_name in fulls:
        key = full_name.lower()
        if key not in FULL_TO_ABBREVIATION:
            FULL_TO_ABBREVIATION[key] = []
        if abbrev not in FULL_TO_ABBREVIATION[key]:
            FULL_TO_ABBREVIATION[key].append(abbrev)


# ── Ambiguous abbreviations ──────────────────────────────────────────────────
# Abbreviations that map to multiple distinct chemicals.  Replacing a specific
# name with one of these forces the agent to disambiguate.

AMBIGUOUS_ABBREVIATIONS: dict[str, list[str]] = {
    "TEA": ["triethylamine", "triethanolamine"],
    "TFA": ["trifluoroacetic acid", "trifluoroacetaldehyde"],
    "DMA": ["dimethylacetamide", "dimethylamine"],
    "PEG": ["polyethylene glycol"],
    "DME": ["1,2-dimethoxyethane", "dimethyl ether"],
    "MTBE": ["methyl tert-butyl ether"],
    "ACN": ["acetonitrile"],
    "IPA": ["isopropanol", "isopropyl alcohol"],
    "NMP": ["N-methyl-2-pyrrolidone"],
    "TBAF": ["tetrabutylammonium fluoride"],
    "TBAI": ["tetrabutylammonium iodide"],
    "TMS": ["trimethylsilyl", "tetramethylsilane"],
    "Ac2O": ["acetic anhydride"],
    "BuOH": ["1-butanol", "2-butanol", "tert-butanol", "isobutanol"],
    "PrOH": ["1-propanol", "2-propanol"],
}


# ── Common / IUPAC name switching ────────────────────────────────────────────
# Maps common trade/trivial names to IUPAC or systematic names.

COMMON_TO_IUPAC: dict[str, str] = {
    "acetone": "propan-2-one",
    "acetic acid": "ethanoic acid",
    "methanol": "methyl alcohol",
    "ethanol": "ethyl alcohol",
    "toluene": "methylbenzene",
    "benzaldehyde": "benzenecarbaldehyde",
    "aniline": "phenylamine",
    "phenol": "hydroxybenzene",
    "styrene": "ethenylbenzene",
    "acetonitrile": "ethanenitrile",
    "pyridine": "azine",
    "piperidine": "azinane",
    "glycine": "2-aminoacetic acid",
    "urea": "carbonyl diamide",
    "formaldehyde": "methanal",
    "chloroform": "trichloromethane",
    "carbon tetrachloride": "tetrachloromethane",
    "dimethyl sulfoxide": "methylsulfinylmethane",
    "triethylamine": "N,N-diethylethanamine",
    "water": "oxidane",
    "hydrogen peroxide": "dihydrogen dioxide",
    "indole": "1H-benzo[b]pyrrole",
    "purine": "7H-imidazo[4,5-d]pyrimidine",
}

IUPAC_TO_COMMON: dict[str, str] = {v.lower(): k for k, v in COMMON_TO_IUPAC.items()}


# ── Salt / hydrate form variation ────────────────────────────────────────────
# Some chemicals appear as hydrochloride salts, hydrates, etc.

SALT_VARIATIONS: dict[str, str] = {
    "sodium hydroxide": "NaOH",
    "potassium hydroxide": "KOH",
    "potassium carbonate": "K2CO3",
    "sodium carbonate": "Na2CO3",
    "sodium bicarbonate": "NaHCO3",
    "cesium carbonate": "Cs2CO3",
    "hydrochloric acid": "HCl",
    "sodium hydride": "NaH",
    "sodium nitrite": "NaNO2",
    "lithium chloride": "LiCl",
    "potassium bicarbonate": "KHCO3",
    "copper(I) iodide": "CuI",
    "copper (I) iodide": "CuI",
    "copper(II) acetate": "Cu(OAc)2",
}

FORMULA_TO_NAME: dict[str, str] = {v.lower(): k for k, v in SALT_VARIATIONS.items()}
