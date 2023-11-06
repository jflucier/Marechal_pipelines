import argparse
from rcsbsearch import TextQuery
from rcsbsearch import rcsb_attributes as attrs
from Bio.PDB import PDBList

def fetch_sinlge_pdb(gene_symbol):


    pdbl = PDBList()
    x = pdbl.retrieve_pdb_file('RFWD3', pdir='/home/jflucier/tmp/pdb', file_format='pdb', overwrite=True, obsolete=False)

    print(x)
    # Create a query for the gene RFWD3 and filter for human entries
    # Create terminals for each query
    # q1 = TextQuery('RFWD3')
    # q2 = attrs.rcsb_entity_source_organism.ncbi_taxonomy_id == [9606]
    #
    # # combined using bitwise operators (&, |, ~, etc)
    # query = q1 & q2  # AND of all queries
    #
    # # Call the query to execute it
    # for pdb_id in query("entry"):
    #     print(pdb_id)
    # Create a text query for the gene name
    # text_query = TextQuery(term="RFWD3", attributes=["rcsb_entity_source_gene.gene_name"])
    #
    # q1 = TextQuery('"RFWD3"')
    # q2 = attrs.rcsb_entity_source_gene.gene_name == "RFWD3"
    # q3 = attrs.rcsb_struct_symmetry.kind == "Global Symmetry"
    # q4 = attrs.rcsb_entry_info.polymer_entity_count_DNA >= 1
    #
    # # Create a search query for human organisms
    # search_query = RcsbSearchQuery()
    # search_query.add_text_query(text_query)
    # search_query.add_entity_source_organism_query("Homo sapiens")
    #
    # # Search for entries and print their IDs
    # search_result = search_query.search(return_type="entry")
    # for entry in search_result:
    #     print(entry.identifier)

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()

    # mandatory
    #argParser.add_argument("-w", "--workdir", help="your working directory", required=True)

    fetch_sinlge_pdb("RFWD3")