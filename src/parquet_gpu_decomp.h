/*
 * parquet_gpu_decomp.h
 *
 * GPU-based Parquet decompression using nvComp
 * ----
 * Copyright 2011-2025 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2025 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#ifndef PARQUET_GPU_DECOMP_H
#define PARQUET_GPU_DECOMP_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/*
 * Compression type enumeration (matching Parquet compression types)
 */
typedef enum {
	PARQUET_COMP_UNCOMPRESSED = 0,
	PARQUET_COMP_SNAPPY = 1,
	PARQUET_COMP_GZIP = 2,
	PARQUET_COMP_LZO = 3,
	PARQUET_COMP_BROTLI = 4,
	PARQUET_COMP_LZ4 = 5,
	PARQUET_COMP_ZSTD = 6,
	PARQUET_COMP_LZ4_RAW = 7
} ParquetCompressionType;

/*
 * Metadata for a compressed column chunk
 */
typedef struct ParquetColumnChunkInfo {
	int				column_index;
	int				compression_type;	/* ParquetCompressionType */
	int64_t			data_page_offset;
	int64_t			dictionary_page_offset;
	int64_t			total_compressed_size;
	int64_t			total_uncompressed_size;
	int64_t			num_values;
} ParquetColumnChunkInfo;

/*
 * Row group decompression info
 */
typedef struct ParquetRowGroupInfo {
	int				row_group_index;
	int				num_columns;
	int64_t			num_rows;
	ParquetColumnChunkInfo *columns;	/* array of column info */
} ParquetRowGroupInfo;

/*
 * Extract metadata from Parquet file for GPU decompression
 *
 * Returns NULL on error, with error message in *p_error_message
 */
ParquetRowGroupInfo *
parquetGetRowGroupInfo(const char *filename,
					   int row_group_index,
					   const int *column_indices,
					   int num_columns,
					   const char **p_error_message);

/*
 * Free ParquetRowGroupInfo structure
 */
void
parquetFreeRowGroupInfo(ParquetRowGroupInfo *info);

/*
 * Read compressed data from Parquet file to GPU memory
 *
 * Reads compressed column chunks directly into GPU memory for decompression
 */
int
parquetReadCompressedToGPU(const char *filename,
						   const ParquetRowGroupInfo *rg_info,
						   void **d_compressed_data,     /* output: GPU pointers array */
						   size_t *compressed_sizes,     /* output: sizes array */
						   const char **p_error_message);

/*
 * Decompress Parquet data on GPU using nvComp
 *
 * Takes compressed data in GPU memory and decompresses it in-place or to
 * a separate output buffer.
 */
int
parquetDecompressOnGPU(const ParquetRowGroupInfo *rg_info,
					   void **d_compressed_data,
					   size_t *compressed_sizes,
					   void **d_uncompressed_data,    /* output: GPU pointers */
					   size_t *uncompressed_sizes,    /* output: sizes */
					   void *cuda_stream,             /* CUstream cast to void* */
					   const char **p_error_message);

#ifdef __cplusplus
}
#endif

#endif /* PARQUET_GPU_DECOMP_H */
