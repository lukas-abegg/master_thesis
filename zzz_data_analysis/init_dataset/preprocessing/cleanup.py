import re
import unicodedata


# trim and unicode transformation
def normalize_sentence(s, do_lower_case, remove_brackets_with_inside, apply_regex_rules):
    if do_lower_case:
        s = s.lower()

    s = s.strip()
    s = unicode_to_ascii(s)
    s = re.sub(r"([.!?])", r" \1", s)

    if remove_brackets_with_inside:
        s = re.sub("[\(\[].*?[\)\]]", "", s)

    s = re.sub(r"[^a-zA-Z0-9%,.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()

    if apply_regex_rules:
        s = prenormalize(s)
        s = map_regex_concepts(s)
    return s

# ----------------------------------------------------------------
# Sentence Normalization, RegEx based replacement
# ----------------------------------------------------------------


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def map_regex_concepts(token):
    """replaces abbreviations matching simple REs, e.g. for numbers, percentages, gene names, by class tokens"""
    for regex, repl in regex_concept_dict:
        if regex.findall(token):
            return repl

    return token


regex_concept_dict = [
    # biomedical
    (re.compile("\w+inib$"), "_chemical_"),
    (re.compile("\w+[ui]mab$"), "_chemical_"),
    (re.compile("->|-->"), "_replacement_"),
    # (re.compile("^(PFS|pfs)$"), "progression-free survival"),

    # number-related concepts
    (re.compile("^[Pp]([=<>≤≥]|</?=|>/?=)\d"), "_p_val_"),
    (re.compile("^((\d+-)?year-old|y\.?o\.?)$"), "_age_"),
    (re.compile("^~?-?\d*[·.]?\d+--?\d*[·.]?\d+$"), "_range_"),
    (re.compile("[a-zA-Z]?(~?[=<>≤≥]|</?=|>/?=)\d?|^(lt|gt|geq|leq)$"), "_ineq_"),
    (re.compile("^~?\d+-fold$"), "_n_fold_"),
    (re.compile("^~?\d+/\d+$|^\d+:\d+$"), "_ratio_"),
    (re.compile("^~?-?\d*[·.]?\d*%$"), "_percent_"),
    (re.compile("^~?\d*(("
                "(kg|\d+g|mg|ug|ng)|"
                "(\d+m|cm|mm|um|nm)|"
                "(\d+l|ml|cl|ul|mol|mmol|nmol|mumol|mo))/?)+$"), "_unit_"),
    # abbreviation starting with letters and containing nums
    (re.compile("^[Rr][Ss]\d+$|"
                "^[Rr]\d+[A-Za-z]$"), "_mutation_"),
    (re.compile("^[a-zA-Z]\w*-?\w*\d+\w*$"), "_abbrev_"),
    # time
    (re.compile("^([jJ]an\.(uary)?|[fF]eb\.(ruary)?|[mM]ar\.(ch)?|"
                "[Aa]pr\.(il)?|[Mm]ay\.|[jJ]un\.(e)?|"
                "[jJ]ul\.(y)?|[aA]ug\.(ust)?|[sS]ep\.(tember)?|"
                "[oO]ct\.(ober)?|[nN]ov\.(ember)?|[dD]ec\.(ember)?)$"), "_month_"),
    (re.compile("^(19|20)\d\d$"), "_year_"),
    # numbers
    (re.compile("^(([Zz]ero(th)?|[Oo]ne|[Tt]wo|[Tt]hree|[Ff]our(th)?|"
                "[Ff]i(ve|fth)|[Ss]ix(th)?|[Ss]even(th)?|[Ee]ight(th)?|"
                "[Nn]in(e|th)|[Tt]en(th)?|[Ee]leven(th)?|"
                "[Tt]went(y|ieth)?|[Tt]hirt(y|ieth)?|[Ff]ort(y|ieth)?|[Ff]ift(y|ieth)?|"
                "[Ss]ixt(y|ieth)?|[Ss]event(y|ieth)?|[Ee]ight(y|ieth)?|[Nn]inet(y|ieth)?|"
                "[Mm]illion(th)?|[Bb]illion(th)?|"
                "[Tt]welv(e|th)|[Hh]undred(th)?|[Tt]housand(th)?|"
                "[Ff]irst|[Ss]econd|[Tt]hird|\d*1st|\d*2nd|\d*3rd|\d+-?th)-?)+$"), "_num_"),
    (re.compile("^~?-?\d+(,\d\d\d)*$"), "_num_"),  # int (+ or -)
    (re.compile("^~?-?((-?\d*[·.]\d+$|^-?\d+[·.]\d*)(\+/-)?)+$"), "_num_"),  # float (+ or -)
    # misc. abbrevs
    (re.compile("^[Vv]\.?[Ss]\.?$|^[Vv]ersus$"), "vs"),
    (re.compile("^[Ii]\.?[Ee]\.?$"), "ie"),
    (re.compile("^[Ee]\.?[Gg]\.?$"), "eg"),
    (re.compile("^[Ii]\.?[Vv]\.?$"), "iv"),
    (re.compile("^[Pp]\.?[Oo]\.?$"), "po")
]


def prenormalize(text):
    """normalize common abbreviations and symbols known to mess with sentence boundary disambiguation"""
    for regex, repl in prenormalize_dict:
        text = regex.sub(repl, text)

    return text


lower_ahead = "(?=[a-z0-9])"
nonword_behind = "(?<=\W)"

prenormalize_dict = [
    # common abbreviations
    (re.compile(nonword_behind + "[Cc]a\.\s" + lower_ahead), "ca  "),
    (re.compile(nonword_behind + "[Ee]\.[Gg]\.\s" + lower_ahead), "e g  "),
    (re.compile(nonword_behind + "[Ee][Gg]\.\s" + lower_ahead), "e g "),
    (re.compile(nonword_behind + "[Ii]\.[Ee]\.\s" + lower_ahead), "i e  "),
    (re.compile(nonword_behind + "[Ii][Ee]\.\s" + lower_ahead), "i e "),
    (re.compile(nonword_behind + "[Aa]pprox\.\s" + lower_ahead), "approx  "),
    (re.compile(nonword_behind + "[Nn]o\.\s" + lower_ahead), "no  "),
    (re.compile(nonword_behind + "[Nn]o\.\s" + "(?=\w\d)"), "no  "),  # no. followed by abbreviation (patient no. V123)
    (re.compile(nonword_behind + "[Cc]onf\.\s" + lower_ahead), "conf  "),
    # scientific writing
    (re.compile(nonword_behind + "et al\.\s" + lower_ahead), "et al  "),
    (re.compile(nonword_behind + "[Rr]ef\.\s" + lower_ahead), "ref  "),
    (re.compile(nonword_behind + "[Ff]ig\.\s" + lower_ahead), "fig  "),
    # medical
    (re.compile(nonword_behind + "y\.o\.\s" + lower_ahead), "y o  "),
    (re.compile(nonword_behind + "yo\.\s" + lower_ahead), "y o "),
    (re.compile(nonword_behind + "[Pp]\.o\.\s" + lower_ahead), "p o  "),
    (re.compile(nonword_behind + "[Ii]\.v\.\s" + lower_ahead), "i v  "),
    (re.compile(nonword_behind + "[Bb]\.i\.\d\.\s" + lower_ahead), "b i d  "),
    (re.compile(nonword_behind + "[Tt]\.i\.\d\.\s" + lower_ahead), "t i d  "),
    (re.compile(nonword_behind + "[Qq]\.i\.\d\.\s" + lower_ahead), "q i d  "),
    (re.compile(nonword_behind + "J\.\s" + "(?=(Cell|Bio|Med))"), "J  "),  # journal
    # bracket complications
    # (re.compile("\.\)\."), " )."),
    # (re.compile("\.\s\)\."), "  )."),
    # multiple dots
    # (re.compile("(\.+\s*\.+)+"), "."),
    # # Typos: missing space after dot; only add space if there are at least two letters before and behind
    # (re.compile("(?<=[A-Za-z]{2})" + "\." + "(?=[A-Z][a-z])"), ". "),
    # whitespace
    (re.compile("\s"), " "),
]
