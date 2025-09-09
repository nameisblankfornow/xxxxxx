import asyncio
import aiohttp
import json
from pydantic import BaseModel
from typing import List, Dict, Any


class DefinitionValidation(BaseModel):
    relation_type: str
    is_accurate: str  # "True" or "False"
    improved_definition: str
    explanation: str


class PredicateMapping(BaseModel):
    mapped_predicate: str
    negated: str = "False"


# Your ChemProt definitions
chemprot_definitions = {
    "agonist": [
        "agonist activator",
        "a chemical substance that binds to and activates certain receptors on cells, causing a biological response",
        "a chemical that activates a receptor to produce a biological response",
        "a drug which under some conditions behaves as an agonist while under other conditions",
        "a substance that fully activates the receptor that it binds to, typically referring to compounds that both bind to receptors and enhance their activity"
    ],
    "part of": [
        "part of",
        "indicates that the chemical is a structural component or subunit of the protein complex"
    ],
    "regulator": [
        "direct regulator",
        "indirect regulator",
        "a general term for compounds that control or influence the activity, expression, or function of proteins or biological processes, encompassing both positive and negative regulation"
    ],
    "upregulator": [
        "indirect upregulator",
        "Usually related to upregulation and words such as 'activate', 'promote' and 'increase activity of'",
        "a compound that increases the expression, production, or activity of genes, proteins, or biological pathways",
        "compound that increases the activity, expression, or function of a protein or biological pathway"
    ],
    "downregulator": [
        "indirect downregulator",
        "substance that decreases or prevents the activity of an enzyme, receptor, or biological process",
        "Usually associated with downregulation and words such as 'inhibitor', 'block' and 'decrease activity of'",
        "compound that decreases the expression, production, or activity of genes, proteins, or biological pathways"
    ],
    "antagonist": [
        "chemical substance that binds to and blocks the activation of certain receptors on cells, preventing a biological response",
        "have affinity but no efficacy for their cognate receptors, and binding will disrupt the interaction and inhibit the function of an agonist"
    ],
    "modulator": [
        "modulator activator",
        "binds to and activates a receptor but is only able to elicit partial efficacy at that receptor or binds to a receptor at a site distinct from the active site and induces a conformational change in the receptor, which alters the affinity of the receptor for the endogenous ligand"
    ],
    "substrate": [
        "product of",
        "substrate product of",
        "related to substrate metabolic relation",
        "the end result molecule produced from an enzymatic reaction where the chemical is the product of the protein's (enzyme's) activity",
        "the molecule upon which an enzyme acts; the starting material in an enzymatic reaction that gets converted to a product",
        "a bidirectional relationship where a chemical can serve as both substrate and product in metabolic processes involving the protein"
    ],
    "cofactor": [
        "a non-protein chemical compound that is required for an enzyme's biological activity",
        "may be organic molecules (coenzymes) or inorganic ions"
    ]
}


async def validate_definition(relation_type: str, definitions: List[str]) -> DefinitionValidation:
    """Validate a single relation type definition with MedGemma"""

    definitions_text = "\n".join([f"- {defn}" for defn in definitions])

    prompt = f'''You are a biomedical expert familiar with the ChemProt dataset for chemical-protein relation extraction.

Evaluate whether the following definitions for the "{relation_type}" relation type are accurate and complete for the ChemProt dataset:

{definitions_text}

Consider:
1. Do these definitions align with standard biomedical usage?
2. Are they specific enough to distinguish from other ChemProt relations?
3. Do they capture the intended semantics of the ChemProt dataset?

Respond with JSON in this exact format:
{{
    "relation_type": "{relation_type}",
    "is_accurate": "True or False",
    "improved_definition": "If not accurate, provide a better definition that aligns with ChemProt dataset intentions. If accurate, repeat the best existing definition.",
    "explanation": "Brief explanation of your assessment"
}}'''

    payload = {
        "model": "alibayram/medgemma:latest",
        "prompt": prompt,
        "stream": False
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post("http://localhost:11434/api/generate", json=payload) as response:
                result = await response.json()
                raw_text = result["response"]
                print(f"Raw response for {relation_type}:")
                print(raw_text[:200] + "..." if len(raw_text) > 200 else raw_text)
                print("-" * 50)

                # Extract JSON
                start = raw_text.find('{')
                end = raw_text.rfind('}') + 1
                if start == -1 or end == 0:
                    print(f"No JSON found for {relation_type}")
                    return None

                json_str = raw_text[start:end]
                data = json.loads(json_str)
                validation = DefinitionValidation(**data)

                return validation

    except Exception as e:
        print(f"Error validating {relation_type}: {e}")
        return None


async def validate_all_definitions():
    """Validate all ChemProt relation definitions"""

    print("Starting ChemProt Definition Validation with MedGemma")
    print("=" * 60)

    results = {}
    improved_definitions = {}

    for relation_type, definitions in chemprot_definitions.items():
        print(f"\nValidating: {relation_type.upper()}")
        print("-" * 30)

        validation = await validate_definition(relation_type, definitions)

        if validation:
            results[relation_type] = validation

            print(f"Validation complete for {relation_type}")
            print(f"Accurate: {validation.is_accurate}")
            print(f"Explanation: {validation.explanation}")

            if validation.is_accurate.lower() == "false":
                improved_definitions[relation_type] = validation.improved_definition
                print(f"Improved definition: {validation.improved_definition}")
            else:
                print(f"Current definitions are acceptable")

        else:
            print(f"Failed to validate {relation_type}")

        await asyncio.sleep(1)

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    accurate_count = 0
    for relation_type, validation in results.items():
        status = "GOOD" if validation.is_accurate.lower() == "true" else "NEEDS IMPROVEMENT"
        print(f"{relation_type.ljust(15)}: {status}")
        if validation.is_accurate.lower() == "true":
            accurate_count += 1

    print(f"\nAccuracy Rate: {accurate_count}/{len(results)} ({accurate_count / len(results) * 100:.1f}%)")

    if improved_definitions:
        print("\n" + "=" * 60)
        print("IMPROVED DEFINITIONS")
        print("=" * 60)

        for relation_type, improved_def in improved_definitions.items():
            print(f"\n{relation_type.upper()}:")
            print(f"  {improved_def}")

    # Create updated definitions in the same format as input
    updated_definitions = {}
    for relation_type, original_defs in chemprot_definitions.items():
        if relation_type in improved_definitions:
            # Use improved definition
            updated_definitions[relation_type] = [improved_definitions[relation_type]]
        else:
            # Keep original definitions
            updated_definitions[relation_type] = original_defs

    # Save the updated definitions in the same format as input
    with open("updated_chemprot_definitions.json", "w") as f:
        json.dump(updated_definitions, f, indent=2)

    # Also save detailed validation results
    validation_output = {
        "validation_results": {k: v.dict() for k, v in results.items()},
        "summary": {
            "total_relations": len(results),
            "accurate_relations": accurate_count,
            "accuracy_rate": accurate_count / len(results) if results else 0
        }
    }

    with open("chemprot_validation_results.json", "w") as f:
        json.dump(validation_output, f, indent=2)

    print(f"\nUpdated definitions saved to updated_chemprot_definitions.json")
    print(f"Validation details saved to chemprot_validation_results.json")

    return results, updated_definitions


async def main():
    try:
        results, updated_definitions = await validate_all_definitions()
        print("\nValidation complete!")
        return results, updated_definitions
    except Exception as e:
        print(f"Error during validation: {e}")
        return None, None


if __name__ == "__main__":
    results, updated_definitions = await main()