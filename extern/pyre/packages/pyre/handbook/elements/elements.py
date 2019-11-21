# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# pull the representation
from .Element import Element

# build the individual elements
hydrogen     = Element(  1, "h",  "hydrogen",       1.00797)
helium       = Element(  2, "he", "helium",         4.00260)
lithium      = Element(  3, "li", "lithium",        6.941)
beryllium    = Element(  4, "be", "beryllium",      9.01218)
boron        = Element(  5, "b",  "boron",         10.81)
carbon       = Element(  6, "c",  "carbon",        12.011)
nitrogen     = Element(  7, "n",  "nitrogen",      14.0067)
oxygen       = Element(  8, "o",  "oxygen",        15.9994)
fluorine     = Element(  9, "f",  "fluorine",      18.9984)
neon         = Element( 10, "ne", "neon",          20.179)
sodium       = Element( 11, "na", "sodium",        22.9898)
magnesium    = Element( 12, "mg", "magnesium",     24.305)
aluminum     = Element( 13, "al", "aluminum",      26.9815)
silicon      = Element( 14, "si", "silicon",       28.0855)
phosphorus   = Element( 15, "p",  "phosphorus",    30.9738)
sulfur       = Element( 16, "s",  "sulfur",        32.06)
chlorine     = Element( 17, "cl", "chlorine",      35.453)
argon        = Element( 18, "ar", "argon",         39.948)
potassium    = Element( 19, "k",  "potassium",     39.0983)
calcium      = Element( 20, "ca", "calcium",       40.08)
scandium     = Element( 21, "sc", "scandium",      44.9559)
titanium     = Element( 22, "ti", "titanium",      47.88)
vanadium     = Element( 23, "v",  "vanadium",      50.9415)
chromium     = Element( 24, "cr", "chromium",      51.996)
manganese    = Element( 25, "mn", "manganese",     54.9380)
iron         = Element( 26, "fe", "iron",          55.847)
cobalt       = Element( 27, "co", "cobalt",        58.9332)
nickel       = Element( 28, "ni", "nickel",        58.69)
copper       = Element( 29, "cu", "copper",        63.546)
zinc         = Element( 30, "zn", "zinc",          65.39)
gallium      = Element( 31, "ga", "gallium",       69.72)
germanium    = Element( 32, "ge", "germanium",     72.59)
arsenic      = Element( 33, "as", "arsenic",       74.9216)
selenium     = Element( 34, "se", "selenium",      78.96)
bromine      = Element( 35, "br", "bromine",       79.904)
krypton      = Element( 36, "kr", "krypton",       83.80)
rubidium     = Element( 37, "rb", "rubidium",      85.4678)
strontium    = Element( 38, "sr", "strontium",     87.62)
yttrium      = Element( 39, "y",  "yttrium",       88.9059)
zirconium    = Element( 40, "zr", "zirconium",     91.224)
niobium      = Element( 41, "nb", "niobium",       92.9064)
molybdenum   = Element( 42, "mo", "molybdenum",    95.94)
technetium   = Element( 43, "tc", "technetium",    98.)
ruthenium    = Element( 44, "ru", "ruthenium",    101.07)
rhodium      = Element( 45, "rh", "rhodium",      102.906)
palladium    = Element( 46, "pd", "palladium",    106.42)
silver       = Element( 47, "ag", "silver",       107.868)
cadmium      = Element( 48, "cd", "cadmium",      112.41)
indium       = Element( 49, "in", "indium",       114.82)
tin          = Element( 50, "sn", "tin",          118.71)
antimony     = Element( 51, "sb", "antimony",     121.75)
tellurium    = Element( 52, "te", "tellurium",    127.60)
iodine       = Element( 53, "i",  "iodine",       126.905)
xenon        = Element( 54, "xe", "xenon",        131.29)
cesium       = Element( 55, "cs", "cesium",       132.905)
barium       = Element( 56, "ba", "barium",       137.33)
lanthanum    = Element( 57, "la", "lanthanum",    138.906)
cerium       = Element( 58, "ce", "cerium",       140.12)
praseodymium = Element( 59, "pr", "praseodymium", 140.908)
neodymium    = Element( 60, "nd", "neodymium",    144.24)
promethium   = Element( 61, "pm", "promethium",   145.)
samarium     = Element( 62, "sm", "samarium",     150.36)
europium     = Element( 63, "eu", "europium",     151.96)
gadolinium   = Element( 64, "gd", "gadolinium",   157.25)
terbium      = Element( 65, "tb", "terbium",      158.925)
dysprosium   = Element( 66, "dy", "dysprosium",   162.50)
holmium      = Element( 67, "ho", "holmium",      164.930)
erbium       = Element( 68, "er", "erbium",       167.26)
thulium      = Element( 69, "tm", "thulium",      168.934)
ytterbium    = Element( 70, "yb", "ytterbium",    173.04)
lutetium     = Element( 71, "lu", "lutetium",     174.967)
hafnium      = Element( 72, "hf", "hafnium",      178.49)
tantalum     = Element( 73, "ta", "tantalum",     180.948)
tungsten     = Element( 74, "w",  "tungsten",     183.85)
rhenium      = Element( 75, "re", "rhenium",      186.207)
osmium       = Element( 76, "os", "osmium",       190.2)
iridium      = Element( 77, "ir", "iridium",      192.22)
platinum     = Element( 78, "pt", "platinum",     195.08)
gold         = Element( 79, "au", "gold",         196.967)
mercury      = Element( 80, "hg", "mercury",      200.59)
thallium     = Element( 81, "tl", "thallium",     204.383)
lead         = Element( 82, "pb", "lead",         207.2)
bismuth      = Element( 83, "bi", "bismuth",      208.980)
polonium     = Element( 84, "po", "polonium",     209.)
astatine     = Element( 85, "at", "astatine",     210.)
radon        = Element( 86, "rn", "radon",        222.)
francium     = Element( 87, "fr", "francium",     223.)
radium       = Element( 88, "ra", "radium",       226.025)
actinium     = Element( 89, "ac", "actinium",     227.028)
thorium      = Element( 90, "th", "thorium",      232.038)
protactinium = Element( 91, "pa", "protactinium", 231.036)
uranium      = Element( 92, "u",  "uranium",      238.029)
neptunium    = Element( 93, "np", "neptunium",    237.048)
plutonium    = Element( 94, "pu", "plutonium",    244.)
americium    = Element( 95, "am", "americium",    243.)
curium       = Element( 96, "cm", "curium",       247.)
berkelium    = Element( 97, "bk", "berkelium",    247.)
californium  = Element( 98, "cf", "californium",  251.)
einsteinium  = Element( 99, "ei", "einsteinium",  252.)
fermium      = Element(100, "fm", "fermium",      257.)
mendelevium  = Element(101, "md", "mendelevium",  258.)
nobelium     = Element(102, "no", "nobelium",     259.)
lawrencium   = Element(103, "lw", "lawrencium",   269.)


# the atomic number index
elements = [
    hydrogen, helium, lithium, beryllium, boron, carbon, nitrogen, oxygen,
    fluorine, neon, sodium, magnesium, aluminum, silicon, phosphorus, sulfur,
    chlorine, argon, potassium, calcium, scandium, titanium, vanadium, chromium,
    manganese, iron, cobalt, nickel, copper, zinc, gallium, germanium, arsenic,
    selenium, bromine, krypton, rubidium, strontium, yttrium, zirconium, niobium,
    molybdenum, technetium, ruthenium, rhodium, palladium, silver, cadmium, indium,
    tin, antimony, tellurium, iodine, xenon, cesium, barium, lanthanum, cerium,
    praseodymium, neodymium, promethium, samarium, europium, gadolinium, terbium,
    dysprosium, holmium, erbium, thulium, ytterbium, lutetium, hafnium, tantalum,
    tungsten, rhenium, osmium, iridium, platinum, gold, mercury, thallium, lead, bismuth,
    polonium, astatine, radon, francium, radium, actinium, thorium, protactinium,
    uranium, neptunium, plutonium, americium, curium, berkelium, californium, einsteinium,
    fermium, mendelevium, nobelium, lawrencium
    ]


# end of file
