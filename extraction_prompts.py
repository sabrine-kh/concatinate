# Prompts for material property extraction and other connector attributes
# Dictionary-based matching approach - LLM should find best match from available values

# --- Material Properties ---

MATERIAL_PROMPT = """
Find the best match for Material Filling from the document context.

Available values: ["none", "GF", "CF", "(GB+GF)"]

Instructions:
- Look for material filling additives in the document
- Common additives: GF (glass-fiber), GB (glass-balls), MF (mineral-fiber), T (talcum)
- Match to the closest available value
- If no fillers mentioned, use "none"
- If multiple additives, use "(GB+GF)" format if applicable

Output format: {"Material Filling": "best_match_from_dictionary"}
"""

MATERIAL_NAME_PROMPT = """
Find the best match for Material Name from the document context.

Available values: ["PA66", "PBT", "PA", "Silicone Rubber", "PA6", "Plastics", "PP", "PA+SPS", "PA12", "PET", "PA66+PA6", "PC"]

Instructions:
- Look for the primary polymer material in the document
- Remove additives/fillers from composite names (PA66-GF30 → PA66)
- Match to the closest available value
- For blends, use combined format if available (PA66+PA6)
- If uncertain, use "NOT FOUND"

Output format: {"Material Name": "best_match_from_dictionary"}
"""

# --- Physical / Mechanical Attributes ---

PULL_TO_SEAT_PROMPT = """
Find the best match for Pull-To-Seat requirement from the document context.

Available values: ["No", "Yes"]

Instructions:
- Look for pull-to-seat, pull-back, or tug-lock mechanisms
- Check for pre-inserted terminals or tool-free insertion
- If pull action is required for terminal seating → "Yes"
- If no pull action or alternative methods → "No"

Output format: {"Pull-To-Seat": "best_match_from_dictionary"}
"""

GENDER_PROMPT = """
Find the best match for Gender from the document context.

Available values: ["female", "male"]

Instructions:
- Look for connector gender indicators: "Plug", "Header" → male
- "Receptacle", "Socket" → female
- Check internal contact types: pins → male function, sockets → female function
- Manufacturer nomenclature takes priority over internal contacts
- If unclear, use "NOT FOUND"

Output format: {"Gender": "best_match_from_dictionary"}
"""

HEIGHT_MM_PROMPT = """
Find the connector height in millimeters from the document context.

Instructions:
- Look for height measurements in Y-axis or total height
- Include CPA/TPA locked position adjustments if specified
- For round connectors, use diameter as height
- Return numerical value or "999" if not found

Output format: {"Height [MM]": "numerical_value_or_999"}
"""

LENGTH_MM_PROMPT = """
Find the connector length in millimeters from the document context.

Instructions:
- Look for length measurements in Z-axis from mating face to rear
- Include CPA/TPA locked position adjustments if specified
- Return numerical value or "999" if not found

Output format: {"Length [MM]": "numerical_value_or_999"}
"""

WIDTH_MM_PROMPT = """
Find the connector width in millimeters from the document context.

Instructions:
- Look for width measurements in X-axis
- For round connectors, use diameter
- Include TPA/CPA locked position adjustments if specified
- Return numerical value or "NOT FOUND" if not found

Output format: {"Width [MM]": "numerical_value_or_NOT_FOUND"}
"""

NUMBER_OF_CAVITIES_PROMPT = """
Find the best match for Number of Cavities from the document context.

Available values: ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "12", "13", "14", "16", "18", "19", "20", "23", "24", "26", "27", "30", "31", "32", "35", "38", "46", "47", "52", "53", "60", "63", "64", "136"]

Instructions:
- Look for cavity count, position count, or "way" indicators
- Check part number suffixes (-2C, -4P, -6W)
- Match to the closest available value
- If not found, use "999"

Output format: {"Number of Cavities": "best_match_from_dictionary"}
"""

NUMBER_OF_ROWS_PROMPT = """
Find the best match for Number of Rows from the document context.

Available values: ["0", "1", "2", "4", "7", "9", "24"]

Instructions:
- Look for row-based structure descriptions
- Check for grid arrangements (e.g., 4x6 grid → 4 rows)
- Match to the closest available value
- If no row structure, use "0"

Output format: {"Number of Rows": "best_match_from_dictionary"}
"""

MECHANICAL_CODING_PROMPT = """
Find the best match for Mechanical Coding from the document context.

Available values: ["None", "A", "B", "C", "D", "E", "F", "G", "I", "Z", "1", "2", "5", "III", "No naming", "Neutral", "X", "II", "V"]

Instructions:
- Look for explicit coding letters: "Coding A", "Coding B", etc.
- Check for "neutral coding", "0-position", "universal coding"
- Look for "no mechanical coding", "not keyed", "not polarized"
- Match to the closest available value
- If not found, use "9999"

Output format: {"Mechanical Coding": "best_match_from_dictionary"}
"""

# --- Color & Appearance ---

COLOUR_PROMPT = """
Find the best match for Colour from the document context.

Available values: ["000 bk", "101 nt", "111 ye", "222 og", "333 rd", "353 pk", "444 vt", "555 bu", "666 gn", "777 gy", "888 bn", "999 wh"]

Instructions:
- Look for color descriptions or color codes
- Match color names to codes: black→"000 bk", natural→"101 nt", yellow→"111 ye", etc.
- For multi-color designs, use "multi"
- If not found, use "NOT FOUND"

Output format: {"Colour": "best_match_from_dictionary"}
"""

COLOUR_CODING_PROMPT = """
Find the best match for Colour Coding from the document context.

Available values: ["None", "Red", "Blue", "Orange", "Natural", "Black", "Pink", "White", "Violet"]

Instructions:
- Look for color-coded mechanical coding components (CPA, TPA, coding keys)
- Check if coding components differ from housing color
- Look for "X denotes variant Y" statements
- If no color coding or all components match housing → "None"
- Match to the closest available color name

Output format: {"Colour Coding": "best_match_from_dictionary"}
"""

# --- Sealing & Environmental ---

WORKING_TEMPERATURE_PROMPT = """
Find the working temperature range from the document context.

Available Max values: ["40.0000", "80.0000", "85.0000", "100.000", "105.000", "120.000", "125.000", "130.000", "135", "140.000", "150.000", "155.000", "-1"]
Available Min values: ["-65.0000", "-55.0000", "-40.0000", "-30.0000", "-20.0000", "-1"]

Instructions:
- Look for temperature ranges (e.g., "-40°C to 125°C")
- Extract max and min temperatures
- Match to closest available values
- If not found, use "-1" for missing value

Output format: {"Working Temperature": "/max_value/min_value"}
"""

HOUSING_SEAL_PROMPT = """
Find the best match for Housing Seal from the document context.

Available values: ["none", "interface seal", "radial seal"]

Instructions:
- Look for "Radial Seal", "Interface Seal", or "Ring Seal"
- Check if seals refer to connector-to-counterpart interface
- If no sealing mentioned → "none"
- "Ring Seal" maps to "radial seal"

Output format: {"Housing Seal": "best_match_from_dictionary"}
"""

WIRE_SEAL_PROMPT = """
Find the best match for Wire Seal from the document context.

Available values: ["none", "single wire seal", "Mat seal", "Silicone family seal", "family seal"]

Instructions:
- Look for individual wire seals per cavity → "single wire seal"
- Look for unified sealing element → "Mat seal"
- Look for "gel seal", "silicone family seal" → "Silicone family seal"
- If no sealing mentioned → "none"
- If not found, use "9999"

Output format: {"Wire Seal": "best_match_from_dictionary"}
"""

SEALING_PROMPT = """
Find the best match for Sealing status from the document context.

Available values: ["unsealed", "sealed"]

Instructions:
- Look for IP codes: IPx0 → "unsealed", IPx4+ → "sealed"
- Check for "waterproof", "dustproof", "sealed", "gasket"
- Check for "unsealed", "no sealing"
- If not found, use "NOT FOUND"

Output format: {"Sealing": "best_match_from_dictionary"}
"""

SEALING_CLASS_PROMPT = """
Find the Sealing Class from the document context.

Available values: ["IPx0", "IPx7", "IPx9K", "IPx6", "IPx4", "IPx8", "IPx5", "not defined", "IPx9K,IPx6", "IPx9K,IPx7", "IPx9K,IPx9K", "IPx6,IPx7", "IPx7,IPx9K", "IPx7,IPx6"]

Instructions:
- Look for IP codes in the document
- Match exact IP codes found
- For multiple codes, use combined format if available
- If no IP rating → "not defined"

Output format: {"Sealing Class": "best_match_from_dictionary"}
"""

# --- Terminals & Connections ---

CONTACT_SYSTEMS_PROMPT = """
Find the best match for Contact Systems from the document context.

Available values: ["TAB 1.8", "0.64", "MCP 2.8", "MLK 1.2", "MQS 0.64", "SLK 2.8", "HF", "070", "GT 2.8", "MTS 0.64", "NG 1.8", "2.3", "BOX 2.8", "QKK 2.8", "RH 0.64", "CTS 1.5", "NanoMQS", "MCON 1.2", "HSD", "RK", "YESC 1.5", "MCP 1.5K", "HCT4", "HPCS 2.8", "2.8", "040", "SPT 4.8", "090 HW", "AMPSEAL", "MOD", "ST", "CONI1 1.6", "Econoseal 1.5", "MCP 1.2", "TAB 1.2", "FASTON 6.3", "M800", "GET 0.64", "MATE-N-LOK", "025 TH", "MPQ 2.8", "MAK 8", "MAK 2.8", "TAB 1.5", "DIA 3.6", "DIA 9.0", "DIA 6.0", "DIA 3.0", "TAB 1.6", "QKK 4.8", "FS 2.8", "FS 1.2", "US 2.8x0.8", "TAB 2.8", "TAB 4.8", "TAB 9.5", "3.5", "MCP 6.3", "MX 1.5", "1.5", "1.2", "QKK 1.2", "MLK 1.2 Sm", "MCP 1.5", "MQS 1.5", "MQS 0.64 CB"]

Instructions:
- Look for contact system families mentioned
- Check for terminal part numbers that map to systems
- Match to exact system names found
- For multiple systems, use comma-separated list

Output format: {"Contact Systems": "best_match_from_dictionary"}
"""

TERMINAL_POSITION_ASSURANCE_PROMPT = """
Find the best match for Terminal Position Assurance from the document context.

Available values: ["None", "1", "2", "undefined_to do not use"]

Instructions:
- Look for "TPA", "Terminal Position Assurance", "Anti-Backout"
- Check if TPA is preassembled (not requiring assembly)
- Count number of TPAs if multiple
- If assembly required → "0"
- If no TPA → "None"

Output format: {"Terminal Position Assurance": "best_match_from_dictionary"}
"""

CONNECTOR_POSITION_ASSURANCE_PROMPT = """
Find the best match for Connector Position Assurance from the document context.

Available values: ["No", "Yes"]

Instructions:
- Look for "CPA", "Connector Position Assurance", "Anti-Backout"
- Check for secondary locking mechanisms
- If CPA mentioned → "Yes"
- If no CPA or explicit denial → "No"

Output format: {"Connector Position Assurance": "best_match_from_dictionary"}
"""

CLOSED_CAVITIES_PROMPT = """
Find the best match for Name of Closed Cavities from the document context.

Available values: ["none", "2,3", "4-7,14-17", "4-5,10,14-15,17,19"]

Instructions:
- Look for closed cavity numbers or ranges
- Check for "blocked positions", "plugged cavities"
- If all cavities open → "none"
- If specific numbers mentioned, match to available patterns
- If not found, use "none"

Output format: {"Name of Closed Cavities": "best_match_from_dictionary"}
"""

# --- Assembly & Type ---

PRE_ASSEMBLED_PROMPT = """
Find the best match for Pre-assembled status from the document context.

Available values: ["No", "Yes"]

Instructions:
- Look for "delivered as assembly", "requires disassembly"
- Check if full connector assembly needs breakdown for production
- If disassembly required → "Yes"
- If components preassembled but no full disassembly needed → "No"
- If not found, use "NOT FOUND"

Output format: {"Pre-assembled": "best_match_from_dictionary"}
"""

CONNECTOR_TYPE_PROMPT = """
Find the best match for Type of Connector from the document context.

Available values: ["Standard", "Antenna", "Contact Carrier", "HSD / USB / HDMI", "Airbag / Squib", "IDC", "Bulb holder", "Relay holder"]

Instructions:
- Look for explicit type mentions: "HSD", "USB", "Antenna", etc.
- Check application context: relay holder, bulb holder, etc.
- For general-purpose connectors → "Standard"
- Match to closest available type

Output format: {"Type of Connector": "best_match_from_dictionary"}
"""

SET_KIT_PROMPT = """
Find the best match for Set/Kit status from the document context.

Available values: ["No", "Yes"]

Instructions:
- Look for "Set", "Kit" explicitly mentioned
- Check if accessories have separate part numbers → "No"
- Check if accessories included under same part number → "Yes"
- If not found, use "NOT FOUND"

Output format: {"Set/Kit": "best_match_from_dictionary"}
"""

# --- Specialized Attributes ---

HV_QUALIFIED_PROMPT = """
Find the best match for HV Qualified status from the document context.

Available values: ["No", "Yes"]

Instructions:
- Look for voltage > 60V and HV context
- Check for "HV-qualified", "HV-certified", "HV-connector"
- Check for orange color and HV safety features
- If ≤60V → "No"
- If >60V with HV context → "Yes"

Output format: {"HV Qualified": "best_match_from_dictionary"}
"""