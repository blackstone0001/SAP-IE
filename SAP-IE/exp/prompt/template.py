# jc 23.11.28

template_ner_natural = (
    "### Extract the entities and their types in the given sentence.\n"
    "\tAnswer only in the form of: (entity_text, entity_type).\n"
    # "Warp each element with double quotation marks.\n"
    "\tThe given sentence is: \"{sentence}\"\n"
    # "The answer is: "
)

template_re_natural = (
    "### Extract the head entity, tail entity and the relation between them in the given sentence.\n"
    "\tAnswer only in the form of: (head entity, relation, tail entity).\n"
    "\tThe given sentence is: \"{sentence}\"\n"
    # "The answer is: "
)

template_pure_re_natural = (
    "### Extract the relation between the head entity and tail entity in the given sentence.\n"
    "\tAnswer only in the form of: (head entity, relation, tail entity).\n"
    "\tThe given sentence is: \"{sentence}\"\n"
    # "The answer is: "
)

template_ner_python = (
    "def named_entity_recognition(input_text):\n"
    "\t\"\"\" extract named entities from the input_text . \"\"\"\n"
    "\tinput_text = \"{sentence}\"\n"
    "\tentity_list = []\n"
    "\t# extracted named entities\n"
)

template_re_python = (
    "def relation_extraction(input_text):\n"
    "\t\"\"\" extract the relations of named entities from the input text. \"\"\"\n"
    "\tinput_text = \"{sentence}\"\n"
    "\tentity_relation_list = []\n"
    "\t# extracted relations\n"
)

template_base = {
    'ner_natural': template_ner_natural,
    're_natural': template_re_natural,
    'pure-re_natural': template_pure_re_natural,
    'ner_python': template_ner_python,
    're_python': template_re_python,
    'pure-re_python': template_re_python,
}

# template_tail = {
#     'code-ner': "\tentity_list.append({{{{\"text\": \"{text}\", \"type\": \"{type}\"}}}})\n",
#     'code-re': "\tentity_relation_list.append({{{{\"rel_type\": \"{rel_type}\", \"ent1_text\": \"{ent1_text}\", \"ent2_text\": \"{ent2_text}\"}}}})\n",
#     'code-pure-re': "\tentity_relation_list.append({{{{\"rel_type\": \"{rel_type}\", \"ent1_text\": \"{ent1_text}\", \"ent2_text\": \"{ent2_text}\"}}}})\n",
# }
