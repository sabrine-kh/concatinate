import os
import streamlit as st
import wandb
from pages.chatbot import (
    find_relevant_markdown_chunks,
    format_markdown_context,
    format_context,
    generate_sql_from_query,
    find_relevant_attributes_with_sql,
    llm_choose_tool,
    get_groq_chat_response,
    leoni_attributes_schema_for_main_loop,
    llm
)
from sentence_transformers import SentenceTransformer, util
from supabase import create_client
import pandas as pd
import re
# Add BLEU and ROUGE imports
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity

# --- Ground truth ---
ground_truth = [
  {
    "question": "Number Of Fuse-Circuits",
    "answer": "For fuses with 1 fuse circuit it's 1. A higher number of fuse circuits is only related to multifuses. The number describes the amount of fuse circuits in a multifuse (e.g. 5 circuits)."
  },
  {
    "question": "What are the rules for the 'Strip Length [mm]' attribute for electric contacts?",
    "answer": "- Use the value specified on the supplier drawing.\n- If only max and min are given, enter their average (e.g. (3.75 + 4.25)/2 = 4.00 mm).\n- If only max or min is given, use that value.\n- If no supplier data exists, calculate approximately: ≤ 1 mm² → S = X + 1 mm; 1 mm² < X ≤ 16 mm² → S = X + 2 mm; > 16 mm² → S = X + 3 mm.\n- If the wire size spans a boundary (e.g. 0.5–1.5 mm²), choose the average."
  },
  {
  "question": "What is the Type Of Inductor?",
  "answer": "Chip inductor: designed for SMD mounting on a PCB\nCoil: wire-wound inductor\nOne core double choke: single core with two independent wire coils\nRF inductor: spaced windings for high-frequency applications\nRing choke: (not defined in the doc, placeholder)\nFilter, Ferrit, CAN-choke: listed under RF inductors without detailed definitions"
  },
  {
    "question": "The grease consists of which material?",
    "answer": "Grease is a semisolid lubricant, generally a soap emulsified with mineral or vegetable oil."
  },
  {
    "question": "The sealant is used for what?",
    "answer": "A sealant is used to ensure sealing of the harness on specific spots: as additional material within a grommet; kneaded between cables/wires of a harness."
  },
  {
    "question": "What is the second name of Elongation at break?",
    "answer": "Elongation at break, also known as fracture strain, is the ratio between changed length and initial length after breakage."
  },
  {
    "question": "what are the types Of Granule?",
    "answer": "Plastic will not always be delivered in its finial molding composition granule. For the final composition multiple basis materials, colour batches and additives can be mixed. This attribute describes if the granule is a basic material or just a colour batch.\n• Basic material\n• Colour batch"
  },
  {
    "question": "Define LED.",
    "answer": "A light-emitting diode (LED) is a two-lead semiconductor light source, functioning like a pn-junction diode that emits light when forward-biased. Mounting Technology: THT, SMD.Operating Voltage [V]: The LED used will have a voltage drop, specified at the intended operating current. If the voltage is below the threshold or on-voltage no current will flow and the result is an unlit LED. If the voltage is too high the current will go above the maximum rating, overheating and potentially destroying the LED."
  },
  {
    "question": "What is the connection type of relay?",
    "answer": "Plug-in: The relay is inserted into a relay holder; male terminals mate with a holder's female terminals.\nScrewed: Contacts secured via screws, typically for high-current applications.\nSoldering SMD: Surface-mounted device (SMD): glued to the PCB first, then soldered en masse."
  },
  {
    "question": "What is the 'HV Qualified' attribute used for?",
    "answer": "The 'HV Qualified' attribute defines if a part is approved for high voltage application, which is specified as a range greater than 60 V. The attribute is set to 'Yes' ONLY when the documentation indicates this property, or the parts are used in an HV-connector or an HV-assembly. Otherwise, it is 'No'."
  },
  {
    "question": "What rule applies to the 'Max. Working Temperature [°C]' attribute if the 3000-hour temperature is unavailable?",
    "answer": "If the temperature for 3000 hours isn't available in the part's documentation, the temperature of the next bigger time term (e.g., 30000 hours) should be used. If there isn't a time term with defined hours and only the information 'permanent temperature' is available, then this temperature can be used."
  },
  {
    "question": "What value should be entered for numeric attributes if a value is definitely not available?",
    "answer": "For numeric attributes where a value is definitely not available (because it makes no sense, can't be clarified, or doesn't exist), the value '999' should be used. This indicates the attribute has been checked and a correct value is unavailable."
  },
  {
    "question": "How are multi-value attributes ordered in the LEOparts object details page?",
    "answer": "The values of multi-value attributes are sorted alphabetically in the object detail page after the part is checked-in, regardless of the order in which they were entered during editing."
  },
  {
    "question": "What is the 'Structure' attribute for conductors?",
    "answer": "The 'Structure' attribute is a special short key used to identify the overall design structure of a wire. The naming can be done according to different guidelines or norms such as JASO, SAE, and DIN 76722."
  },
  {
    "question": "What is the 'Capillar-Stop-Function' in a wire?",
    "answer": "The 'Capillar-Stop-Function' is a feature in anti-capillary wires and cables that prevents liquids (like water or oil) from flowing through the conductors via capillary action, which could otherwise damage electronic circuitry."
  },
  {
    "question": "What is the difference between a stranded wire and a solid wire?",
    "answer": "A stranded wire is an electrical conductor made of many thin wires (strands), making it easy to bend. A solid wire, also called solid-core or single-strand wire, consists of one single piece of metal wire and is more rigid."
  },
  {
    "question": "How is the 'Cross Section [mm²]' of a stranded wire calculated?",
    "answer": "The cross-sectional area (A) in [mm²] is calculated using the formula: A = (π * d² / 4) * N, where 'd' is the diameter of a single strand in [mm] and 'N' is the number of single strands."
  },
  {
    "question": "When is a cable considered 'High Flexible'?",
    "answer": "A cable is considered 'High Flexible' when the diameter of its wire strands is very small in comparison to the external diameter of the cable, making the wire 'finely stranded'. This attribute is set to 'Yes' if the datasheet indicates this property with wording like 'Flexible' or 'HighFlex'."
  },
  {
    "question": "What are the three main types of wire stranding?",
    "answer": "The three types are: S-Stranding (left-handed, counter-clockwise twist), Z-Stranding (right-handed, clockwise twist), and SZ-Stranding (stranding with changing twist direction to improve layer stability)."
  },
  {
    "question": "What components can a multi cable consist of?",
    "answer": "A multi cable consists of a minimum of two conductors and can include: two or several single wires, wires with a screen, wires with a drain wire, fillers, or a combination of these elements, all surrounded by an additional insulation layer."
  },
  {
    "question": "In a multi cable, are drain wires and screens counted in the 'Number Of Cables' attribute?",
    "answer": "No, the 'Number Of Cables' attribute does not count the drain wire or the screen. Only single cables and other multi cables within the main multi cable are counted."
  },
  {
    "question": "What are the possible values for the 'Type Of Shield' attribute in a multi cable?",
    "answer": "The possible values for the 'Type Of Shield' attribute are: None, Braid, Foil, and Braid and Foil."
  },
  {
    "question": "What is a 'Screen' in the context of multi cables?",
    "answer": "A screen is a sub-classification for multi cables that acts as a Faraday cage to reduce electrical noise and electromagnetic radiation. The material can be Copper (Cu), Aluminum (Al), or Steel."
  },
  {
    "question": "What is the purpose of a 'Drain Wire'?",
    "answer": "A drain wire is an uninsulated wire typically used in shielded cables. It is in contact with the metallic side of the shield foil and is used to ground the shield, providing a path for unwanted electrical noise to be drained away."
  },
  {
    "question": "What are the two types of flat cables described in the document?",
    "answer": "The document describes two types of flat cables: 'exFC – Extruded flat cable', which consists of several flat copper wires arranged in parallel and insulated, and 'FLC – Flat laminated cable'."
  },
  {
    "question": "What is the 'Basic Grid [mm]' for a flat cable?",
    "answer": "The 'Basic Grid' (also known as pitch) for a flat cable is the distance measured in mm between the centers of two successive conducting wires."
  },
  {
    "question": "How is a 'Twisted pair' classified if it consists of more than two wires?",
    "answer": "Parts that consist of twisted wires, even if there are more than two, are classified as 'Twisted pair' because 'twisted' is the main identifying information."
  },
  {
    "question": "What is 'Lay Length' for a twisted pair?",
    "answer": "Lay Length is the distance in mm which a core travels from its starting position in a layer around the cable and back to its original position, essentially completing one full twist."
  },
  {
    "question": "What are the two main types of material used for optical fibers?",
    "answer": "The two main types of material are Glass, for Glass Optical Fiber (GOF), and Polymer, for Polymer Optical Fibers (POF)."
  },
  {
    "question": "When is a 'Braided Conductor' classified as a 'Cable Assembly'?",
    "answer": "A braided conductor is classified as a 'Cable Assembly' if it is assembled with other parts, such as an eyelet. If it has no other parts assembled, it is classified as a Braided Conductor."
  },
  {
    "question": "What is the 'Shape' attribute for a Braided Conductor?",
    "answer": "The 'Shape' attribute is used to describe if the braided conductors are round or flat."
  },
  {
    "question": "How is the 'Length [mm]' of a cable assembly determined?",
    "answer": "The length of a cable assembly is defined as the nominal value of the whole wire, excluding tolerances and housing lengths. If the drawing only provides a length including housings, that nominal length should be used and then rounded to an integer."
  },
  {
    "question": "How is the length of a cable assembly with multiple different wire lengths (e.g., a Y-cable) entered in LEOparts?",
    "answer": "For cable assemblies with multiple wire lengths, the length of every individual wire is entered into the system by using the multi-value field. For example, a Y-cable might have lengths of 400 mm and 700 mm entered separately."
  },
  {
    "question": "What does it mean if the 'Pigtail' attribute for a cable assembly is 'Yes'?",
    "answer": "'Pigtail' is 'Yes' if any additional work, such as cutting, crimping, or adding a connector, must be performed on the cable assembly itself at a Leoni plant before it can be used in the final harness."
  },
  {
    "question": "What defines an 'HV-ASSY' (High-Voltage Assembly)?",
    "answer": "An HV-ASSY is an assembly of high-voltage and high-current cables, typically for hybrid vehicles. A key visual identifier is that they always have an orange outfit."
  },
  {
    "question": "What is a 'CONNECTOR-TERMINAL-ASSY'?",
    "answer": "It is a classification for cable assemblies that have a connector(s) or any kind of electric contact(s) (like terminals or eyelets) and do not fit into other specific categories such as SENSOR, HSD, or USB. It is a renaming of the old 'STANDARD' type."
  },
  {
    "question": "When is a cable assembly classified as 'SENSOR'?",
    "answer": "An assembly is classified as 'SENSOR' only if it includes an actual converter or sensor as part of the assembly. A cable that is simply used to connect to an external sensor is classified as a 'CONNECTOR-TERMINAL-ASSY', not a sensor assembly."
  },
  {
    "question": "What is the exception for classifying a part as a 'Connector' versus an 'electronic module'?",
    "answer": "If a housing includes 2 or more electronic components (e.g., a PCB and a resistor), it is classified as an 'electronic module'. A standard connector accepts terminals but may have limited or no integrated electronic components."
  },
  {
    "question": "How are the X, Y, and Z dimensions of a connector defined?",
    "answer": "The dimensions are based on a coordinate system relative to the mating face. X-axis is the longer dimension of the mating face (Width). Y-axis is the shorter dimension of the mating face (Height). Z-axis is the third dimension, representing the connector's Length."
  },
  {
    "question": "What is 'Connector Position Assurance' (CPA)?",
    "answer": "Connector Position Assurance (CPA) is an additional protection feature to ensure that a connector is placed correctly to its counterpart and to prevent it from being removed unintentionally. It can only be activated when the connector is in the right position."
  },
  {
    "question": "What information should be used for the 'Contact Systems' attribute of a connector?",
    "answer": "The 'Contact Systems' attribute should define the terminal family that is used in the connector (e.g., MQS, BTL1.5, RK). It should not be the name of the connector family itself."
  },
  {
    "question": "What is the rule for the 'Gender' attribute for a connector with different cavity types?",
    "answer": "If a connector contains different cavities designed for both male and female terminals, its gender is defined as 'hybrid'."
  },
  {
    "question": "How are the dimensions of a connector measured if it includes a TPA or CPA?",
    "answer": "The dimensions (Length, Height, Width) are measured as if the connector is fully assembled, meaning the TPA (Terminal Position Assurance) or CPA (Connector Position Assurance) is in its final, locked position."
  },
  {
    "question": "What value should be used for 'Mechanical Coding' if a drawing shows the coding feature but provides no name for it?",
    "answer": "If the mechanical coding is only shown graphically on the drawing without any specific name or identifier given, the value to be used is 'no naming'."
  },
  {
    "question": "Under what two conditions can we talk about 'Colour Coding' for a connector?",
    "answer": "To be able to talk about 'Colour Coding', two conditions must be met: 1. The connectors must have a mechanical coding. 2. The connectors must have a different or additional color on individual parts of the housing (e.g., inner housing, TPA, or rear cap)."
  },
  {
    "question": "How is the 'Number Of Cavities' determined for a connector with numbered cavities?",
    "answer": "The 'Number of Cavities' is the highest number shown on the housing itself, regardless of how many cavities might be closed or blocked, but only if the cavities have numerations."
  },
  {
    "question": "When is the 'Number Of Rows' for a connector set to 0?",
    "answer": "The 'Number of Rows' is set to 0 if the connector does not have straight rows, if the cavity sizes are different (e.g., a mix of tab size 1.5 and 2.8), or for 1-cavity connectors."
  },
  {
    "question": "What does 'Pre-Assembled = Yes' mean for a connector?",
    "answer": "This attribute defines if the connector is delivered as an assembly (e.g., with a TPA, CPA, or lever) which has to be disassembled in our production in order to be used."
  },
  {
    "question": "What is 'Terminal Position Assurance' (TPA)?",
    "answer": "Terminal Position Assurance (TPA) is a component that provides a secondary locking mechanism for a terminal within a connector housing. It guarantees the terminal is fully seated and prevents it from backing out, supplementing the primary lock."
  },
  {
    "question": "What is a 'Contact Carrier' type of connector?",
    "answer": "A 'Contact Carrier' is a type of housing part that is designed to be assembled with terminals and then needs a separate base housing to form a complete connector assembly. It cannot be used as a standalone single-piece connector."
  },
  {
    "question": "What are the different options for 'Wire Seal' on a connector?",
    "answer": "The different possibilities for wire sealing available on a connector are: Single wire seal, Injected, Mat seal (which includes 'gel family seal' and 'silicone family seal'), and None."
  },
  {
    "question": "What does 'Set/Kit = Yes' mean for a connector?",
    "answer": "'Set/Kit = Yes' means the connector is delivered with one LEONI part number but includes separate, non-preassembled accessories like a cover, lever, or TPA. All these loose pieces are handled under the same Leoni part number and will be assembled at the production site."
  },
  {
    "question": "When does a 'Housing Part' have its 'Electrical Function' attribute set to 'Yes'?",
    "answer": "The 'Electrical Function' attribute is set to 'Yes' if the housing part provides any electrical functionality, for example by having an integrated metal part like a shortcut bridge, a PCB-terminal, an LED, or a jumper."
  },
  {
    "question": "What is the function of a 'Strain-relief' type Housing Part?",
    "answer": "A 'Strain-relief' is an independent housing part that is assembled onto connectors to fix electrical cables. Its function is to relieve stress from the cable connections to prevent them from breaking due to tensile forces."
  },
  {
    "question": "What is a 'Seal holder' type of housing part and how does it differ from a cavity mask?",
    "answer": "A 'Seal holder' only fixes one or more seals, but it does not protect a cavity from dust or water. A cavity mask, on the other hand, can be used to close unused cavities, sometimes in addition to a mat seal."
  },
  {
    "question": "What defines an 'Isolator' housing part and what material is it made of?",
    "answer": "An 'Isolator' is a housing part used to isolate two electrical conductors from each other, for instance, separating a wire from a shield within the same cavity in an HV connector. These parts are never made of metal."
  },
  {
    "question": "What is a 'Terminal stabilizer' housing part?",
    "answer": "A 'Terminal stabilizer' is an additional part used to center and/or stabilize the terminals of a connector, protecting them from bending. It is mostly used in male connectors and is also known as a 'male blade stabilizer'."
  },
  {
    "question": "What is the purpose of a 'Cap or cover'?",
    "answer": "A cap or cover is used to protect a connector, other parts like battery lugs or eyelets, or a wire connection spot against environmental conditions."
  },
  {
    "question": "How do you differentiate between a Cap and a Cover for a connector?",
    "answer": "A Cap is assembled on the interface (mating) side of a connector to protect the terminals inside. A Cover is used on the wire side of a connector to protect the wires where they enter the housing."
  },
  {
    "question": "What are the rules for the 'Strip Length [mm]' attribute for electric contacts?",
    "answer": "- Use the value specified on the supplier drawing.\n- If only max and min are given, enter their average (e.g. (3.75 + 4.25)/2 = 4.00 mm).\n- If only max or min is given, use that value.\n- If no supplier data exists, calculate an approximate value: ≤ 1 mm² → S = X + 1 mm; 1-16 mm² → S = X + 2 mm; > 16 mm² → S = X + 3 mm, where X is the crimp width."
  },
  {
    "question": "What are the different shapes ('Terminal Shape') for a terminal?",
    "answer": "The general types of terminal shapes are: Box, Round, Spring, Flat, B-shape, Keyhole, and Hybrid (for terminals with more than one shape)."
  },
  {
    "question": "What does the 'Wire Size Range From [mm²]' attribute describe and what is the rule for the '>' symbol?",
    "answer": "It describes the minimal wire cross section [mm²] suitable for the terminal. If the range is given with a '>' symbol (e.g., > 2.50 mm²), the attribute should be filled according to the rule: > X.xx mm² + 0.01 -> X.x1 mm² (e.g., > 2.50 mm² becomes 2.51 mm²)."
  },
  {
    "question": "What are the three levels of 'Sealability' for a terminal?",
    "answer": "The three levels are: Sealable (if it's assembled with a sealing part or intended for use with a single wire seal), Unsealable (if the insulation crimp is not designed for a seal), and sealable or unsealable (if the terminal can be used with or without a seal)."
  },
  {
    "question": "How do you determine if a terminal is a 1-part or 2-part component?",
    "answer": "A terminal is 2-part if there is a separate box (spring) at the contact zone, which often has a different material than the terminal body. A 1-part terminal does not have this separate component and will have only one material description, like 'copper alloy'."
  },
  {
    "question": "What are the different types of 'Wire Connection Type' for a terminal?",
    "answer": "The technology used to fit a terminal to a wire can be: Crimped, Soldered, Welded, Screwed, or Pressed."
  },
  {
    "question": "What is the 'Primary Lock' on a terminal?",
    "answer": "The 'Primary Lock' is the feature that initially locks the terminal into the connector housing. This can be a metallic 'Locking Lance' on the terminal itself, or for 'Clean Body' terminals, it's achieved by plastic lances inside the housing."
  },
  {
    "question": "What is the 'Fixing Type' for a Battery Lug?",
    "answer": "The 'Fixing Type' describes how a battery lug is fixed to the battery pole. The two types are: screwed or clamped."
  },
  {
    "question": "When is the 'Multi Connection' attribute for a battery lug set to 'yes'?",
    "answer": "The 'Multi Connection (yes / no)' attribute is set to 'yes' if more than one wire connection point is available on the battery lug."
  },
  {
    "question": "What is the rule for classifying a part as an Eyelet versus a Busbar?",
    "answer": "If the part has one hole (for a stud) plus one or more connection areas for a wire, it is an Eyelet. If it has a stud or more than one hole, it is a Busbar."
  },
  {
    "question": "What are the main types of Eyelets?",
    "answer": "The main types of eyelets are: Standard Eyelet, Squeezable Eyelet, Tubular Eyelet, Multi wire connection spot eyelet, and a special type with an integrated Fuse."
  },
  {
    "question": "What does 'Captive Accessory' mean for an eyelet?",
    "answer": "A 'Captive Accessory' is 'Yes' if the eyelet has any attached accessory, such as a nut, that is used to assemble the eyelet in a vehicle."
  },
  {
    "question": "What is 'Protection Against Torsion' for an eyelet?",
    "answer": "This feature indicates that the eyelet is built in a way that it cannot be turned around on a bolt or against another eyelet during normal usage, ensuring it stays in its optimal position."
  },
  {
    "question": "What is a 'Double Female' terminal?",
    "answer": "A double female is a terminal with two female contact bodies and no direct wire connection like a crimp. It acts as a bridge between two male terminals."
  },
  {
    "question": "What are the wire types allowed for an 'HF terminal'?",
    "answer": "Allowed wire types for HF terminals are listed in a table and include specific coaxial cable types like RTK 031, RG59, RG58, RG179, RG178, RG174, HSD, and a general 'HV' category."
  },
  {
    "question": "What is a 'Lead frame'?",
    "answer": "A lead frame is a metallic part, often stamped or etched, that conducts electricity. It's commonly used in electrical components or chips to connect the die to the external leads."
  },
  {
    "question": "What types of 'PCB Terminal' exist?",
    "answer": "The types of PCB terminals are: Crossterminal, Doublefemale, Female, Forkterminal, and Male."
  },
  {
    "question": "What is a 'Shielding contact' and where is it used?",
    "answer": "A shielding contact is a part used to connect a cable's shield to an electrical contact. It is commonly used in antenna, HV, HSD, and Ethernet connectors."
  },
  {
    "question": "What is the difference between a Splice and a Termination Splice?",
    "answer": "A standard splice connects wires, allowing them to pass through. A 'Termination Splice' terminates two or more wires at a closed end; wires enter from one side, but cannot exit the other."
  },
  {
    "question": "What is a 'Terminal Busbar'?",
    "answer": "A terminal busbar is a special delivery form where multiple terminals are kept together on a single strip, allowing them to be handled and used as a kind of busbar."
  },
  {
    "question": "What is a Single Wire Seal (SWS) and what is its shape?",
    "answer": "A single wire seal (SWS) is used to seal the space between a single wire and a connector cavity. It is crimped to the wire with the terminal. All single wire seals are considered to be round in shape."
  },
  {
    "question": "How is the 'Borehole Diameter [mm]' of a single wire seal calculated if not given on the drawing?",
    "answer": "If the borehole diameter is not given on the drawing or datasheet, it can be calculated by taking the minimum outer diameter of the allowed cable and subtracting 0.2 mm."
  },
  {
    "question": "What is a 'Mat seal' and when does it have a 'Skin'?",
    "answer": "A mat seal is a rubber mat that covers the entire cavity port area of a connector. The attribute 'Skin' is set to 'yes' if the mat seal is delivered with a thin, closed skin over the holes, which the terminals must break through during insertion."
  },
  {
    "question": "What is the purpose of a 'Cavity plug'?",
    "answer": "A cavity plug is used to seal empty, unused cavities of a connector against environmental factors like water and dust."
  },
  {
    "question": "What is a 'Radial seal'?",
    "answer": "A radial seal is used to seal the space between two round components, such as between connector parts or between a corrugated tube and an adapter. The main sealing effect is on the inner and outer diameters."
  },
  {
    "question": "What is an 'Interface seal'?",
    "answer": "An interface seal is used to ensure tightness between the mating faces of a connector and its counterpart."
  },
  {
    "question": "What is the general rule to classify a fixing element as a 'Bolt' or a 'Screw'?",
    "answer": "If the element has both a head for torquing and a thread, it is a screw. If it lacks a head (just a thread) or lacks a thread (just a head), it is a bolt."
  },
  {
    "question": "What does the 'Property Class' of a bolt or screw indicate?",
    "answer": "The property class consists of two numbers that describe the material strength. The first number relates to the nominal tensile strength, and the second number defines the ratio of the yield strength to the tensile strength."
  },
  {
    "question": "What are the different types of screw drives?",
    "answer": "The standard screw drive shapes are: crosshead, hexagon socket-head, slot-screw, square screw, Torx, and Knurl."
  },
  {
    "question": "What is the purpose of a 'Bush'?",
    "answer": "A bush is a mechanical component, often a hollow cylinder or sleeve, that can be used for various purposes like providing a bearing surface, spacing, or as a support sleeve for shielding contacts in connectors."
  },

  {
    "question": "How is a 'Tie lead-through + Edge' clip attached?",
    "answer": "This type of clip is designed to be attached to the edge of a sheet metal panel or plastic part."
  },
  {
    "question": "What is a 'Tie lead-through + Arrowhead' clip?",
    "answer": "This is a clip that combines a guide for a tie (lead-through) with an arrowhead-style fastener that pushes into a hole to secure it."
  },
  {
    "question": "What does the 'Serration side' attribute for a cable tie describe?",
    "answer": "This attribute describes whether the serrations (teeth) of the cable tie are on the inside or the outside of the loop."
  },
 
  {
    "question": "What is the function of a 'P-Clamp'?",
    "answer": "P-clamps, named for their 'P' shape, are used to lock cables, tubes, or sleeves in place by bracketing them to a surface."
  },

  {
    "question": "What are the different types of 'Rubber Boot'?",
    "answer": "Rubber boots are protective coverings. The types are based on what they cover: Connector, Terminal (including non-shrinking insulation sleeves), Eyelet, and Battery Lug."
  },
  {
    "question": "When is an End Cap classified as a Heat-shrink sleeve?",
    "answer": "If an end cap needs to be shrunken to fit or seal, it should be classified under 'Heat-shrink sleeve' rather than 'End Cap'."
  },
  {
    "question": "What does the 'Adaptable' attribute for an Adapter signify?",
    "answer": "The 'Adaptable' attribute is 'Yes' if a further adapter can be fixed to the existing one, which is usually indicated by the presence of external or internal ribs."
  },
  {
    "question": "What is the difference between an 'Open' and 'Closed' adapter design?",
    "answer": "An 'Open' design adapter can be opened (e.g., with a latch) to place wires inside, while a 'Closed' design is a solid piece, and each wire must be threaded through it."
  },
  {
    "question": "What is a 'Cable routing' part?",
    "answer": "A cable routing is a part, often a plastic conduit or channel, used to conduct, fixate, and/or protect cables and wires in a harness or vehicle. It can be a single part or an assembly."
  },
  {
    "question": "What is the default color for Crepe-Paper tape?",
    "answer": "If not given differently on the data sheet, the default color for Crepe-Paper tape is beige (171)."
  },
  {
    "question": "What is the 'Abrasion Resistance [LV312-1]' attribute for tape?",
    "answer": "It is an attribute that classifies a tape's resistance to rubbing or friction according to the LV312-1 standard, with classes from A (low resistance) to G (extremely high resistance)."
  },
  {
    "question": "What is the 'Adhesive' on a tape made of?",
    "answer": "The adhesive can be made of Acrylic basis, Natural Rubber, Synthetic Rubber, or it can be a 'Hook' type material (like Velcro). 'None' is used if there is no adhesive."
  },
  {
    "question": "What is the difference between Felt and Non-woven (Fleece) tape?",
    "answer": "Both are textiles made by matting fibers. Felt is a specific kind of non-woven, often made with wool/animal hair, and has its own customs tariff code (5602). Non-woven (Fleece) is a broader category (code 5603). The supplier must be asked if the type is unclear."
  },
  {
    "question": "What does the length attribute '0' signify for a 'Tube or Sleeve'?",
    "answer": "If a tube or sleeve is supplied on a role (continuous length) rather than in pre-cut pieces, its length attribute is set to '0'."
  },
  {
    "question": "How is a corrugated tube with the 'Self-closing' lockable feature designed?",
    "answer": "'Self-closing' means the edges of the slitted tube overlap slightly to ensure that no wire will be damaged between the edges when it closes around the bundle."
  },
  {
    "question": "How is the 'Insulation Thickness [mm]' of a heat-shrink sleeve with adhesive defined?",
    "answer": "For a heat-shrink sleeve with an adhesive liner, the insulation thickness is the total value of the jacket thickness after heating plus the adhesive liner thickness."
  },
  {
    "question": "What are the main types of 'Production support tool'?",
    "answer": "The main types are: Extraction tool (for unlocking TPA, etc.), Safety knife, Installation Aid, and Mixing tube (for fluid materials)."
  },
  {
    "question": "What are the types of 'Adhesive' based on adhesion method?",
    "answer": "The types are: fast setting adhesive, hot melt adhesive, and two component adhesive."
  },
 
  {
    "question": "What are the different types of 'Marking Material'?",
    "answer": "The types are: Printer Label, Marker, Clip, Ident sleeve, Plate, Tape, RFID Label, Amperage Identification, and RFID Token."
  },
  {
    "question": "What is the difference between a BJT and a Darlington transistor?",
    "answer": "A BJT (Bipolar Junction Transistor) is a single semiconductor device used for amplification. A Darlington transistor is a compound structure made of two BJT transistors connected in a way that the current amplified by the first is amplified further by the second, providing very high current gain."
  }
]
# Helper: get chatbot answer (wraps existing logic, does not change it)
def get_chatbot_answer(question):
    # Use the same logic as the chatbot UI
    tool_choice = llm_choose_tool(question, llm)
    relevant_attribute_rows = []
    relevant_markdown_chunks = []
    context_was_found = False
    if tool_choice == "sql":
        generated_sql = generate_sql_from_query(question, leoni_attributes_schema_for_main_loop)
        if generated_sql:
            relevant_attribute_rows = find_relevant_attributes_with_sql(generated_sql)
            context_was_found = bool(relevant_attribute_rows)
    relevant_markdown_chunks = find_relevant_markdown_chunks(question, limit=3)
    if relevant_markdown_chunks:
        context_was_found = True
    attribute_context = format_context(relevant_attribute_rows)
    markdown_context = format_markdown_context(relevant_markdown_chunks)
    combined_context = ""
    if relevant_attribute_rows:
        combined_context += f"**Database Attributes Information:**\n{attribute_context}\n\n"
    if relevant_markdown_chunks:
        combined_context += f"**Documentation/Standards Information:**\n{markdown_context}\n\n"
    if not combined_context:
        combined_context = "No relevant information found in the knowledge base (attributes or documentation)."
    prompt_for_llm = f"Context:\n{combined_context}\n\nUser Question: {question}\n"
    return get_groq_chat_response(prompt_for_llm, context_provided=context_was_found)

# Helper: Generate questions from answer using Qwen LLM
def generate_questions_from_answer(answer, llm, n=3):
    prompt = (
        f"Given the following answer, generate {n} different questions that this answer could be a response to.\n\n"
        f"Answer: \"{answer}\"\nQuestions:\n1."
    )
    response = llm.invoke(prompt) if hasattr(llm, 'invoke') else llm(prompt)
    if hasattr(response, "content"):
        text = response.content
    else:
        text = str(response)
    questions = []
    for line in text.split('\n'):
        line = line.strip()
        if line and (line[0].isdigit() and line[1] in ['.', ')']):
            q = line[2:].strip()
            if q:
                questions.append(q)
    if not questions:
        questions = [l.strip() for l in text.split('\n') if l.strip()]
    return questions[:n]

# Helper: Compute response relevancy
def response_relevancy(user_question, chatbot_answer, llm, embedding_model, n=3):
    generated_questions = generate_questions_from_answer(chatbot_answer, llm, n=n)
    user_emb = embedding_model.encode([user_question])[0]
    scores = []
    for q in generated_questions:
        q_emb = embedding_model.encode([q])[0]
        sim = cosine_similarity([user_emb], [q_emb])[0][0]
        scores.append(sim)
    return sum(scores) / len(scores) if scores else 0.0

# Authenticate wandb using Streamlit secrets
os.environ["WANDB_API_KEY"] = st.secrets["WANDB_API_KEY"]

st.title("Document Search Evaluation with wandb")
st.write("This page evaluates your document search using the provided ground truth and logs results to wandb.")

# Move the chatbot vs ground truth evaluation button to the very top
if st.button("Run Chatbot vs Ground Truth Evaluation"):
    try:
        wandb.init(project="chatbot-vs-gt-eval")
        wandb_initialized = True
    except Exception as e:
        st.error(f"Failed to initialize wandb: {e}")
        wandb_initialized = False
    
    st.subheader("🤖 Chatbot Output vs Ground Truth Evaluation")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    hits = 0
    results = []
    progress = st.progress(0)
    # Metrics setup
    bleu_scores = []
    rouge_l_scores = []
    context_precisions = []
    context_recalls = []
    context_f1s = []
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    smooth = SmoothingFunction().method1
    SIMILARITY_THRESHOLD = 0.4  # Use a variable for easy adjustment
    response_relevancies = []
    for idx, item in enumerate(ground_truth):
        question = item["question"]
        expected_answer = item["answer"]
        try:
            chatbot_answer = get_chatbot_answer(question)
            emb_gt = model.encode(expected_answer, convert_to_tensor=True)
            emb_cb = model.encode(chatbot_answer, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(emb_gt, emb_cb).item()
            hit = similarity > SIMILARITY_THRESHOLD
            hits += int(hit)
            # BLEU and ROUGE-L with input checks
            if expected_answer.strip() and chatbot_answer.strip() and len(expected_answer.split()) > 1 and len(chatbot_answer.split()) > 1:
                bleu = sentence_bleu(
                    [expected_answer.split()],
                    chatbot_answer.split(),
                    smoothing_function=smooth
                )
                rouge = scorer.score(expected_answer, chatbot_answer)
                rouge_l = rouge['rougeL'].fmeasure
            else:
                bleu = 0.0
                rouge_l = 0.0
            bleu_scores.append(bleu)
            rouge_l_scores.append(rouge_l)
            # Standard Context Precision: compare chatbot answer to ground truth in chunks
            # For chatbot, treat the answer as a single chunk, but for standard, you would use multiple chunks
            # Here, let's simulate by splitting the chatbot answer into sentences (as pseudo-chunks)
            # Better chunking: filter out short chunks and take only top 3 meaningful chunks
            all_sentences = [s.strip() for s in re.split(r'[.!?]', chatbot_answer) if s.strip()]
            # Filter out very short chunks (likely formatting artifacts)
            meaningful_chunks = [chunk for chunk in all_sentences if len(chunk) > 15 and not chunk.isdigit() and not chunk in ['g', 'e', 'etc']]
            # Take only the top 3 most meaningful chunks
            answer_chunks = meaningful_chunks[:3] if meaningful_chunks else all_sentences[:3]
            
            # Get chunks from Supabase for this question
            supabase_chunks = find_relevant_markdown_chunks(question, limit=10)
            num_supabase_chunks = len(supabase_chunks) if supabase_chunks else 0
            
            relevant_chunks = 0
            for chunk in answer_chunks:
                emb_chunk = model.encode(chunk, convert_to_tensor=True)
                sim_chunk = util.pytorch_cos_sim(emb_gt, emb_chunk).item()
                if sim_chunk > SIMILARITY_THRESHOLD:
                    relevant_chunks += 1
            context_precision = relevant_chunks / len(answer_chunks) if answer_chunks else 0
            context_precisions.append(context_precision)
            # Context Recall and F1 remain as before (since we have only one answer per question)
            context_recall = context_precision
            context_recalls.append(context_recall)
            context_f1 = context_precision
            context_f1s.append(context_f1)
            response_relevancy_score = response_relevancy(question, chatbot_answer, llm, model)
            response_relevancies.append(response_relevancy_score)
            results.append({
                "question": question,
                "expected_answer": expected_answer,
                "chatbot_answer": chatbot_answer,
                "similarity": similarity,
                "hit": hit,
                "bleu": bleu,
                "rouge_l": rouge_l,
                "context_precision": context_precision,
                "context_recall": context_recall,
                "context_f1": context_f1,
                "num_supabase_chunks": num_supabase_chunks,
                "num_answer_chunks": len(answer_chunks),
                "response_relevancy": response_relevancy_score,
            })
            if wandb_initialized:
                wandb.log({
                    "question": question,
                    "expected_answer": expected_answer,
                    "chatbot_answer": chatbot_answer,
                    "similarity": similarity,
                    "hit": hit,
                    "bleu": bleu,
                    "rouge_l": rouge_l,
                    "context_precision": context_precision,
                    "context_recall": context_recall,
                    "context_f1": context_f1,
                    "num_supabase_chunks": num_supabase_chunks,
                    "num_answer_chunks": len(answer_chunks),
                    "response_relevancy": response_relevancy_score,
                })
            with st.expander(f"{'✅' if hit else '❌'} Q{idx+1}: {question[:50]}...", expanded=False):
                st.markdown(f"**Question:** {question}")
                st.markdown(f"**Expected Answer:** {expected_answer}")
                st.markdown(f"**Chatbot Answer:** {chatbot_answer[:500]}{'...' if len(chatbot_answer) > 500 else ''}")
                st.markdown(f"**Similarity Score:** {similarity:.3f}")
                st.markdown(f"**Hit:** {'✅ Yes' if hit else '❌ No'}")
                st.markdown(f"**BLEU:** {bleu:.3f}")
                st.markdown(f"**ROUGE-L:** {rouge_l:.3f}")
                # Debug output for BLEU/ROUGE inputs
                st.markdown(f"<details><summary>BLEU/ROUGE Inputs</summary>"
                            f"<ul><li>Expected: <code>{expected_answer}</code></li>"
                            f"<li>Chatbot: <code>{chatbot_answer}</code></li></ul></details>", unsafe_allow_html=True)
                
                # Detailed metric calculations
                st.markdown("### 📊 Metric Calculations")
                
                # Supabase chunks information
                st.markdown("**Supabase Retrieval:**")
                st.markdown(f"- Number of chunks retrieved from database: {num_supabase_chunks}")
                if supabase_chunks:
                    st.markdown("**Retrieved Chunks Sources:**")
                    for i, chunk in enumerate(supabase_chunks[:3]):  # Show first 3 chunks
                        source = chunk.get('source', 'Unknown')
                        page = chunk.get('page', 'N/A')
                        st.markdown(f"- Chunk {i+1}: Source: {source}, Page: {page}")
                    if len(supabase_chunks) > 3:
                        st.markdown(f"- ... and {len(supabase_chunks) - 3} more chunks")
                
                # Context Precision calculation details
                st.markdown("**Context Precision Calculation:**")
                st.markdown(f"- Total meaningful answer chunks (top 3): {len(answer_chunks)}")
                st.markdown(f"- Relevant chunks (similarity > {SIMILARITY_THRESHOLD}): {relevant_chunks}")
                st.markdown(f"- Context Precision = {relevant_chunks}/{len(answer_chunks)} = {context_precision:.3f}")
                
                # Show chunk-by-chunk analysis
                st.markdown("**Chunk Analysis (Top 3 Meaningful Chunks):**")
                for i, chunk in enumerate(answer_chunks):
                    emb_chunk = model.encode(chunk, convert_to_tensor=True)
                    sim_chunk = util.pytorch_cos_sim(emb_gt, emb_chunk).item()
                    status = "✅ Relevant" if sim_chunk > SIMILARITY_THRESHOLD else "❌ Not Relevant"
                    st.markdown(f"- Chunk {i+1}: '{chunk[:60]}{'...' if len(chunk) > 60 else ''}' → Similarity: {sim_chunk:.3f} → {status}")
                
                st.markdown(f"**Standard Context Precision:** {context_precision:.3f}")
                st.markdown(f"**Context Recall:** {context_recall:.3f}")
                st.markdown(f"**Context F1:** {context_f1:.3f}")
                st.markdown(f"**Response Relevancy:** {response_relevancy_score:.3f}")
        except Exception as e:
            st.error(f"Error for question '{question}': {e}")
            if wandb_initialized:
                wandb.log({"error": f"{question}: {e}"})
        progress.progress((idx + 1) / len(ground_truth))
    accuracy = hits / len(ground_truth)
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0.0
    context_precisions = [r["context_precision"] for r in results]
    context_recalls = [r["context_recall"] for r in results]
    context_f1s = [r["context_f1"] for r in results]
    similarities = [r["similarity"] for r in results]
    avg_context_precision = sum(context_precisions) / len(context_precisions) if context_precisions else 0.0
    avg_context_recall = sum(context_recalls) / len(context_recalls) if context_recalls else 0.0
    avg_context_f1 = sum(context_f1s) / len(context_f1s) if context_f1s else 0.0
    avg_answer_correctness = sum(similarities) / len(similarities) if similarities else 0.0
    avg_response_relevancy = sum(response_relevancies) / len(response_relevancies) if response_relevancies else 0.0
    st.success(f"🤖 **Chatbot Evaluation Complete!** Final Accuracy: {accuracy:.1%} ({hits}/{len(ground_truth)} questions answered correctly)")
    metrics = {
        "Accuracy": f"{accuracy:.1%}",
        "Avg BLEU": f"{avg_bleu:.3f}",
        "Avg ROUGE-L": f"{avg_rouge_l:.3f}",
        "Avg Standard Context Precision": f"{avg_context_precision:.3f}",
        "Avg Context Recall": f"{avg_context_recall:.3f}",
        "Avg Context F1": f"{avg_context_f1:.3f}",
        "Avg Answer Correctness Score": f"{avg_answer_correctness:.3f}",
        "Avg Response Relevancy": f"{avg_response_relevancy:.3f}",
        "Total Questions": f"{len(ground_truth)}",
        "Hits": f"{hits}"
    }
    st.table(pd.DataFrame(metrics.items(), columns=['Metric', 'Value']))
    if wandb_initialized:
        wandb.log({
            "accuracy": accuracy,
            "avg_bleu": avg_bleu,
            "avg_rouge_l": avg_rouge_l,
            "avg_context_precision": avg_context_precision,
            "avg_context_recall": avg_context_recall,
            "avg_context_f1": avg_context_f1,
            "avg_answer_correctness": avg_answer_correctness,
            "avg_response_relevancy": avg_response_relevancy,
            "total_questions": len(ground_truth),
            "hits": hits
        })
        wandb.finish()
    st.info("📊 **Results logged to wandb** - Check your dashboard for detailed analytics and charts.")

with st.sidebar:
    st.markdown("<h2 style='color:white;'>Navigation</h2>", unsafe_allow_html=True)
    if st.button("🏠 Home", key="home_btn"):
        st.switch_page("app.py")
    if st.button("💬 Chat with Leoparts", key="chat_btn"):
        st.switch_page("pages/chatbot.py")
    if st.button("📄 Extract a new Part", key="extract_btn"):
        st.switch_page("pages/extraction_attributs.py")
    if st.button("🆕 New conversation", key="new_conv_btn"):
        st.session_state.messages = []
        st.session_state.last_part_number = None
        st.rerun()
    if st.button("📊 Evaluate Doc Search", key="eval_btn"):
        st.switch_page("pages/evaluate_doc_search.py")
