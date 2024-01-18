#!/usr/bin/env Rscript

library(pathfindR)
library("optparse")

# pan <- Sys.getenv("RSTUDIO_PANDOC")
# Sys.setenv(RSTUDIO_PANDOC=pan)
# print(paste0("Set RSTUDIO_PANDOC variable to: ", pan))

option_list = list(
  make_option(c("-i", "--gene_lst"), type="character", default=NULL, 
              help="gene list path", metavar="character"),
  make_option(c("-o", "--out"), type="character", default=NULL, 
              help="output dir", metavar="character")
); 

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

if (is.null(opt$gene_lst)){
  print_help(opt_parser)
  stop("Input file to gene list must be specified.n", call.=FALSE)
}

if (is.null(opt$out)){
  print_help(opt_parser)
  stop("Output directory must be specified.n", call.=FALSE)
}

f=opt$gene_lst
o=opt$out

if (file.exists(o)){
  stop("Output directory already exists. Please delete before running.n", call.=FALSE)
}
  
g_list= read.csv(
  f, 
  header = TRUE,
  sep="\t",
  na.strings = "", 
  stringsAsFactors=FALSE
)

gsets_list <- get_gene_sets_list(
  source = "Reactome",
  species = "Homo sapiens"
)


# out=paste(o,"/out",sep='')
output_df <- run_pathfindR(
  g_list,
  output_dir = o,
  # gene_sets = "Reactome",
  custom_genes = gsets_list$gene_sets,
  custom_descriptions = gsets_list$descriptions,
  min_gset_size = 5,
  max_gset_size = 500,
  n_processes = 4,
  #pin_name_path = "STRING"
  pin_name_path = "Biogrid"
)

output_df_clustered <- cluster_enriched_terms(output_df, plot_dend = FALSE, plot_clusters_graph = FALSE)
# plotting only selected clusters for better visualization
selected_clusters <- subset(output_df_clustered[output_df_clustered$Status == "Representative", ], Cluster %in% 1:10)

# output png enrichment chart
out=paste(o,"/enrichment_chart.png",sep='')
png(
  filename = out,
  res = 250,
  width = 8,
  height = 4,
  units = "in"
)
plot(enrichment_chart(selected_clusters, plot_by_cluster = TRUE))
dev.off()

# output png enrichment chart
out=paste(o,"/enrichment_chart.svg",sep='')
svg(
  filename = out,
  width = 8,
  height = 4
)
plot(enrichment_chart(selected_clusters, plot_by_cluster = TRUE))
dev.off()

