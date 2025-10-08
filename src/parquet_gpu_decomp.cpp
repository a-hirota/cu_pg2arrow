/*
 * parquet_gpu_decomp.cpp
 *
 * GPU-based Parquet decompression using nvComp
 * ----
 * Copyright 2011-2025 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2025 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/stat.h>
#include <memory>
#include <string>
#include <vector>

#include <arrow/io/api.h>
#include <parquet/api/reader.h>
#include <parquet/file_reader.h>
#include <parquet/metadata.h>

#include <cuda_runtime.h>
#include <nvcomp.h>
#include <nvcomp/snappy.h>
#include <nvcomp/lz4.h>
#include <nvcomp/gzip.h>
#include <nvcomp/zstd.h>

#include "parquet_gpu_decomp.h"

/* Error reporting */
static __thread char __error_buffer[512];
#define SET_ERROR(fmt, ...) \
	snprintf(__error_buffer, sizeof(__error_buffer), \
			 "[%s:%d] " fmt, __FILE__, __LINE__, ##__VA_ARGS__)

/*
 * parquetGetRowGroupInfo
 *
 * Extract metadata from Parquet file for specified row group and columns
 */
extern "C" ParquetRowGroupInfo *
parquetGetRowGroupInfo(const char *filename,
					   int row_group_index,
					   const int *column_indices,
					   int num_columns,
					   const char **p_error_message)
{
	*__error_buffer = '\0';
	*p_error_message = __error_buffer;

	try {
		/* Open Parquet file */
		std::unique_ptr<parquet::ParquetFileReader> reader =
			parquet::ParquetFileReader::OpenFile(filename, false);

		if (!reader) {
			SET_ERROR("Failed to open Parquet file: %s", filename);
			return NULL;
		}

		/* Get file metadata */
		std::shared_ptr<parquet::FileMetaData> file_metadata = reader->metadata();

		if (row_group_index < 0 || row_group_index >= file_metadata->num_row_groups()) {
			SET_ERROR("Invalid row group index %d (total: %d)",
					  row_group_index, file_metadata->num_row_groups());
			return NULL;
		}

		/* Get row group metadata */
		std::unique_ptr<parquet::RowGroupMetaData> rg_metadata =
			file_metadata->RowGroup(row_group_index);

		/* Allocate result structure */
		ParquetRowGroupInfo *info = (ParquetRowGroupInfo *)malloc(sizeof(ParquetRowGroupInfo));
		if (!info) {
			SET_ERROR("Out of memory");
			return NULL;
		}

		info->row_group_index = row_group_index;
		info->num_rows = rg_metadata->num_rows();
		info->num_columns = num_columns;
		info->columns = (ParquetColumnChunkInfo *)calloc(num_columns,
														 sizeof(ParquetColumnChunkInfo));
		if (!info->columns) {
			free(info);
			SET_ERROR("Out of memory");
			return NULL;
		}

		/* Extract column chunk metadata */
		for (int i = 0; i < num_columns; i++) {
			int col_idx = column_indices[i];

			if (col_idx < 0 || col_idx >= rg_metadata->num_columns()) {
				parquetFreeRowGroupInfo(info);
				SET_ERROR("Invalid column index %d (total: %d)",
						  col_idx, rg_metadata->num_columns());
				return NULL;
			}

			std::unique_ptr<parquet::ColumnChunkMetaData> col_metadata =
				rg_metadata->ColumnChunk(col_idx);

			ParquetColumnChunkInfo *col_info = &info->columns[i];
			col_info->column_index = col_idx;
			col_info->compression_type = (int)col_metadata->compression();
			col_info->data_page_offset = col_metadata->data_page_offset();
			col_info->dictionary_page_offset = col_metadata->dictionary_page_offset();
			col_info->total_compressed_size = col_metadata->total_compressed_size();
			col_info->total_uncompressed_size = col_metadata->total_uncompressed_size();
			col_info->num_values = col_metadata->num_values();
		}

		return info;
	}
	catch (const std::exception &e) {
		SET_ERROR("Exception: %s", e.what());
		return NULL;
	}
}

/*
 * parquetFreeRowGroupInfo
 */
extern "C" void
parquetFreeRowGroupInfo(ParquetRowGroupInfo *info)
{
	if (info) {
		if (info->columns)
			free(info->columns);
		free(info);
	}
}

/*
 * parquetReadCompressedToGPU
 *
 * Read compressed column data directly to GPU memory
 */
extern "C" int
parquetReadCompressedToGPU(const char *filename,
						   const ParquetRowGroupInfo *rg_info,
						   void **d_compressed_data,
						   size_t *compressed_sizes,
						   const char **p_error_message)
{
	*__error_buffer = '\0';
	*p_error_message = __error_buffer;

	try {
		/* Open file using Arrow IO */
		arrow::Result<std::shared_ptr<arrow::io::ReadableFile>> result =
			arrow::io::ReadableFile::Open(filename);

		if (!result.ok()) {
			SET_ERROR("Failed to open file: %s", result.status().ToString().c_str());
			return -1;
		}

		std::shared_ptr<arrow::io::ReadableFile> file = *result;

		/* Read each column chunk */
		for (int i = 0; i < rg_info->num_columns; i++) {
			const ParquetColumnChunkInfo *col_info = &rg_info->columns[i];

			/* Determine starting offset (dictionary page or data page) */
			int64_t start_offset = col_info->data_page_offset;
			if (col_info->dictionary_page_offset > 0 &&
				col_info->dictionary_page_offset < start_offset) {
				start_offset = col_info->dictionary_page_offset;
			}

			size_t read_size = col_info->total_compressed_size;
			compressed_sizes[i] = read_size;

			/* Allocate host buffer for reading */
			void *h_buffer = malloc(read_size);
			if (!h_buffer) {
				SET_ERROR("Out of memory for host buffer (size: %zu)", read_size);
				return -1;
			}

			/* Read compressed data from file */
			arrow::Result<int64_t> read_result =
				file->ReadAt(start_offset, read_size, h_buffer);

			if (!read_result.ok() || *read_result != (int64_t)read_size) {
				free(h_buffer);
				SET_ERROR("Failed to read column %d data", i);
				return -1;
			}

			/* Allocate GPU memory */
			cudaError_t cuda_err = cudaMalloc(&d_compressed_data[i], read_size);
			if (cuda_err != cudaSuccess) {
				free(h_buffer);
				SET_ERROR("cudaMalloc failed: %s", cudaGetErrorString(cuda_err));
				return -1;
			}

			/* Copy to GPU */
			cuda_err = cudaMemcpy(d_compressed_data[i], h_buffer, read_size,
								  cudaMemcpyHostToDevice);
			free(h_buffer);

			if (cuda_err != cudaSuccess) {
				SET_ERROR("cudaMemcpy failed: %s", cudaGetErrorString(cuda_err));
				return -1;
			}
		}

		return 0;
	}
	catch (const std::exception &e) {
		SET_ERROR("Exception: %s", e.what());
		return -1;
	}
}

/*
 * parquetDecompressOnGPU
 *
 * Decompress Parquet column data on GPU using nvComp
 */
extern "C" int
parquetDecompressOnGPU(const ParquetRowGroupInfo *rg_info,
					   void **d_compressed_data,
					   size_t *compressed_sizes,
					   void **d_uncompressed_data,
					   size_t *uncompressed_sizes,
					   void *cuda_stream_ptr,
					   const char **p_error_message)
{
	*__error_buffer = '\0';
	*p_error_message = __error_buffer;

	cudaStream_t stream = (cudaStream_t)cuda_stream_ptr;

	try {
		/* Decompress each column */
		for (int i = 0; i < rg_info->num_columns; i++) {
			const ParquetColumnChunkInfo *col_info = &rg_info->columns[i];

			/* Skip uncompressed columns */
			if (col_info->compression_type == PARQUET_COMP_UNCOMPRESSED) {
				d_uncompressed_data[i] = d_compressed_data[i];
				uncompressed_sizes[i] = compressed_sizes[i];
				continue;
			}

			nvcompStatus_t nvcomp_status;
			size_t temp_size = 0;
			size_t output_size = col_info->total_uncompressed_size;
			void *d_temp = NULL;

			/* Get decompression temp buffer size based on compression type */
			switch (col_info->compression_type) {
				case PARQUET_COMP_SNAPPY:
					nvcomp_status = nvcompBatchedSnappyDecompressGetTempSizeAsync(
						1, output_size,
						nvcompBatchedSnappyDecompressDefaultOpts,
						&temp_size, 0);
					break;

				case PARQUET_COMP_LZ4:
				case PARQUET_COMP_LZ4_RAW:
					nvcomp_status = nvcompBatchedLZ4DecompressGetTempSizeAsync(
						1, output_size,
						nvcompBatchedLZ4DecompressDefaultOpts,
						&temp_size, 0);
					break;

				case PARQUET_COMP_ZSTD:
					nvcomp_status = nvcompBatchedZstdDecompressGetTempSizeAsync(
						1, output_size,
						nvcompBatchedZstdDecompressDefaultOpts,
						&temp_size, 0);
					break;

				case PARQUET_COMP_GZIP:
					nvcomp_status = nvcompBatchedGzipDecompressGetTempSizeAsync(
						1, output_size,
						nvcompBatchedGzipDecompressDefaultOpts,
						&temp_size, 0);
					break;

				default:
					SET_ERROR("Unsupported compression type: %d",
							  col_info->compression_type);
					return -1;
			}

			if (nvcomp_status != nvcompSuccess) {
				SET_ERROR("nvcomp GetTempSize failed for column %d", i);
				return -1;
			}

			/* Allocate temp buffer */
			if (temp_size > 0) {
				cudaError_t err = cudaMalloc(&d_temp, temp_size);
				if (err != cudaSuccess) {
					SET_ERROR("cudaMalloc temp failed: %s", cudaGetErrorString(err));
					return -1;
				}
			}

			/* Allocate output buffer */
			cudaError_t err = cudaMalloc(&d_uncompressed_data[i], output_size);
			if (err != cudaSuccess) {
				if (d_temp) cudaFree(d_temp);
				SET_ERROR("cudaMalloc output failed: %s", cudaGetErrorString(err));
				return -1;
			}

			/* Perform decompression */
			size_t actual_output_size = output_size;

			switch (col_info->compression_type) {
				case PARQUET_COMP_SNAPPY:
					nvcomp_status = nvcompBatchedSnappyDecompressAsync(
						&d_compressed_data[i], &compressed_sizes[i],
						&output_size, &actual_output_size,
						1, d_temp, temp_size,
						&d_uncompressed_data[i],
						nvcompBatchedSnappyDecompressDefaultOpts,
						NULL, stream);
					break;

				case PARQUET_COMP_LZ4:
				case PARQUET_COMP_LZ4_RAW:
					nvcomp_status = nvcompBatchedLZ4DecompressAsync(
						&d_compressed_data[i], &compressed_sizes[i],
						&output_size, &actual_output_size,
						1, d_temp, temp_size,
						&d_uncompressed_data[i],
						nvcompBatchedLZ4DecompressDefaultOpts,
						NULL, stream);
					break;

				case PARQUET_COMP_ZSTD:
					nvcomp_status = nvcompBatchedZstdDecompressAsync(
						&d_compressed_data[i], &compressed_sizes[i],
						&output_size, &actual_output_size,
						1, d_temp, temp_size,
						&d_uncompressed_data[i],
						nvcompBatchedZstdDecompressDefaultOpts,
						NULL, stream);
					break;

				case PARQUET_COMP_GZIP:
					nvcomp_status = nvcompBatchedGzipDecompressAsync(
						&d_compressed_data[i], &compressed_sizes[i],
						&output_size, &actual_output_size,
						1, d_temp, temp_size,
						&d_uncompressed_data[i],
						nvcompBatchedGzipDecompressDefaultOpts,
						NULL, stream);
					break;
			}

			if (d_temp)
				cudaFree(d_temp);

			if (nvcomp_status != nvcompSuccess) {
				SET_ERROR("nvcomp decompression failed for column %d: status=%d",
						  i, nvcomp_status);
				return -1;
			}

			uncompressed_sizes[i] = actual_output_size;
		}

		/* Synchronize stream */
		if (stream) {
			cudaError_t err = cudaStreamSynchronize(stream);
			if (err != cudaSuccess) {
				SET_ERROR("cudaStreamSynchronize failed: %s", cudaGetErrorString(err));
				return -1;
			}
		}

		return 0;
	}
	catch (const std::exception &e) {
		SET_ERROR("Exception: %s", e.what());
		return -1;
	}
}
