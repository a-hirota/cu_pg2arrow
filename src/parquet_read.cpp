/*
 * parquet_read.cc
 *
 * Routines to read Parquet files
 * ----
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include <iostream>
#include <memory>
#include <mutex>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/filesystem/api.h>
#include <arrow/type.h>
#include <parquet/arrow/reader.h>
#include <parquet/api/reader.h>
#include <string>
#include <sys/stat.h>
#include <sys/time.h>
#include <thread>
#include <unistd.h>
#include <cuda_runtime.h>
#include "xpu_common.h"
#include "arrow_defs.h"

/* Forward declaration for cuDF reader (C++ linkage) */
kern_data_store *parquetReadRowGroupCuDF(
	const char *filename,
	int row_group_index,
	const std::vector<int> &column_indices,
	const kern_data_store *kds_head,
	void *(*malloc_callback)(void *malloc_private, size_t malloc_size),
	void *malloc_private,
	const char **p_error_message);

/*
 * Error Reporting
 */
static thread_local	char	__private_error_message[512];
#define __Elog(fmt,...)										\
	snprintf(__private_error_message,						\
			 sizeof(__private_error_message),				\
			 "[error %s:%d] " fmt "\n",						\
			 __basename(__FILE__),__LINE__, ##__VA_ARGS__)	\

/*
 * parquetMetaDataCache
 */
struct parquetMetaDataCache
{
	uint32_t		hash;
	int				refcnt;
	__dlist_node<parquetMetaDataCache> chain;
	struct stat		stat_buf;
	std::string		filename;
	std::shared_ptr<parquet::FileMetaData> metadata;
	/* constructor */
	parquetMetaDataCache(const char *__filename,
						 const struct stat *__stat_buf,
						 uint32_t __hash)
	{
		hash = __hash;
		refcnt = 1;
		chain.owner = this;
		memcpy(&stat_buf, __stat_buf, sizeof(struct stat));
		filename = std::string(__filename);
		metadata = nullptr;		/* to be set caller */
	}
};
using parquetMetaDataCache	= struct parquetMetaDataCache;

#define PQ_HASH_NSLOTS	797
static std::mutex		pq_hash_lock[PQ_HASH_NSLOTS];
static __dlist_node<parquetMetaDataCache> pq_hash_slot[PQ_HASH_NSLOTS];

/*
 * __parquetFileMetaDataHash
 */
static inline uint32_t
__parquetLocalFileHash(dev_t st_dev, ino_t st_ino)
{
	uint64_t	hkey = ((uint64_t)st_dev << 32 | (uint64_t)st_ino);

	hkey += 0x9e3779b97f4a7c15ULL;
	hkey = (hkey ^ (hkey >> 30)) * 0xbf58476d1ce4e5b9ULL;
	hkey = (hkey ^ (hkey >> 27)) * 0x94d049bb133111ebULL;
	return (int64_t)((hkey ^ (hkey >> 31)) & 0xffffffffU);
}

/*
 * parquetPutMetaDataCache
 */
static void
parquetPutMetaDataCache(parquetMetaDataCache *entry)
{
	uint32_t	hindex = entry->hash % PQ_HASH_NSLOTS;

	pq_hash_lock[hindex].lock();
	assert(entry->refcnt > 0);
	if (--entry->refcnt == 0)
	{
		__dlist_delete(&entry->chain);
		delete(entry);
	}
	pq_hash_lock[hindex].unlock();
}

/*
 * __checkParquetFileColumn
 */
static bool
__checkParquetFileColumn(const std::shared_ptr<arrow::Field> &field,
						 const kern_data_store *kds_head, int kds_col_index)
{
	auto	cmeta = &kds_head->colmeta[kds_col_index];
	auto	dtype = field->type();

	switch (dtype->id())
	{
		case arrow::Type::type::NA:
			if (cmeta->attopts.tag == ArrowType__Null)
				return true;
			__Elog("not compatible Null column[%d] TYPE=%s",
				   kds_col_index,
				   ArrowTypeTagAsCString(cmeta->attopts.tag));
			break;
		case arrow::Type::type::BOOL:
			if (cmeta->attopts.tag == ArrowType__Bool)
				return true;
			__Elog("not compatible Bool column[%d] TYPE=%s",
				   kds_col_index,
				   ArrowTypeTagAsCString(cmeta->attopts.tag));
			break;
		case arrow::Type::type::INT8:
		case arrow::Type::type::INT16:
		case arrow::Type::type::INT32:
		case arrow::Type::type::INT64:
			if (cmeta->attopts.tag == ArrowType__Int &&
				cmeta->attopts.integer.bitWidth == dtype->bit_width() &&
				cmeta->attopts.integer.is_signed)
				return true;
			__Elog("not compatible Int column[%d] TYPE=%s bitWidth=%d is_signed=%s",
				   kds_col_index,
				   ArrowTypeTagAsCString(cmeta->attopts.tag),
				   cmeta->attopts.integer.bitWidth,
				   cmeta->attopts.integer.is_signed ? "true" : "false");
			break;
		case arrow::Type::type::UINT8:
		case arrow::Type::type::UINT16:
		case arrow::Type::type::UINT32:
		case arrow::Type::type::UINT64:
			if (cmeta->attopts.tag == ArrowType__Int &&
				cmeta->attopts.integer.bitWidth == dtype->bit_width() &&
				!cmeta->attopts.integer.is_signed)
				return true;
			__Elog("not compatible Uint column[%d] TYPE=%s bitWidth=%d is_signed=%s",
				   kds_col_index,
				   ArrowTypeTagAsCString(cmeta->attopts.tag),
				   cmeta->attopts.integer.bitWidth,
				   cmeta->attopts.integer.is_signed ? "true" : "false");
			break;
		case arrow::Type::type::HALF_FLOAT:
			if (cmeta->attopts.tag == ArrowType__FloatingPoint &&
				cmeta->attopts.floating_point.precision == ArrowPrecision__Half)
				return true;
			__Elog("not compatible HalfFloat column[%d] TYPE=%s precision=%s",
				   kds_col_index,
				   ArrowTypeTagAsCString(cmeta->attopts.tag),
				   ArrowPrecisionAsCString(cmeta->attopts.floating_point.precision));
			break;
		case arrow::Type::type::FLOAT:
			if (cmeta->attopts.tag == ArrowType__FloatingPoint &&
				cmeta->attopts.floating_point.precision == ArrowPrecision__Single)
				return true;
			__Elog("not compatible Float column[%d] TYPE=%s precision=%s",
				   kds_col_index,
				   ArrowTypeTagAsCString(cmeta->attopts.tag),
				   ArrowPrecisionAsCString(cmeta->attopts.floating_point.precision));
			break;
		case arrow::Type::type::DOUBLE:
			if (cmeta->attopts.tag == ArrowType__FloatingPoint &&
				cmeta->attopts.floating_point.precision == ArrowPrecision__Double)
				return true;
			__Elog("not compatible Double column[%d] TYPE=%s precision=%s",
				   kds_col_index,
				   ArrowTypeTagAsCString(cmeta->attopts.tag),
				   ArrowPrecisionAsCString(cmeta->attopts.floating_point.precision));
			break;
		case arrow::Type::type::DECIMAL128:
			if (cmeta->attopts.tag == ArrowType__Decimal &&
				cmeta->attopts.decimal.bitWidth == 128)
			{
				const auto __dtype = std::static_pointer_cast<arrow::Decimal128Type>(dtype);
				if (cmeta->attopts.decimal.precision == __dtype->precision() &&
					cmeta->attopts.decimal.scale     == __dtype->scale())
					return true;
				__Elog("not compatible Decimal column[%d] TYPE=%s precision=%d scale=%d",
					   kds_col_index,
					   ArrowTypeTagAsCString(cmeta->attopts.tag),
					   cmeta->attopts.decimal.precision,
					   cmeta->attopts.decimal.scale);
			}
			else
			{
				__Elog("not compatible Decimal column[%d] TYPE=%s bitWidth=%d",
					   kds_col_index,
					   ArrowTypeTagAsCString(cmeta->attopts.tag),
					   cmeta->attopts.decimal.bitWidth);
			}
			break;
		case arrow::Type::type::STRING:
			if (cmeta->attopts.tag == ArrowType__Utf8)
				return true;
			__Elog("not compatible Utf8 column[%d] TYPE=%s",
				   kds_col_index,
				   ArrowTypeTagAsCString(cmeta->attopts.tag));
			break;
		case arrow::Type::type::LARGE_STRING:
			if (cmeta->attopts.tag == ArrowType__LargeUtf8)
				return true;
			__Elog("not compatible LargeUtf8 column[%d] TYPE=%s",
				   kds_col_index,
				   ArrowTypeTagAsCString(cmeta->attopts.tag));
			break;
		case arrow::Type::type::BINARY:
			if (cmeta->attopts.tag == ArrowType__Binary)
				return true;
			__Elog("not compatible Binary column[%d] TYPE=%s",
				   kds_col_index,
				   ArrowTypeTagAsCString(cmeta->attopts.tag));
			break;
		case arrow::Type::type::LARGE_BINARY:
			if (cmeta->attopts.tag == ArrowType__LargeBinary)
				return true;
			__Elog("not compatible LargeBinary column[%d] TYPE=%s",
				   kds_col_index,
				   ArrowTypeTagAsCString(cmeta->attopts.tag));
			break;
		case arrow::Type::type::FIXED_SIZE_BINARY:
			if (cmeta->attopts.tag == ArrowType__FixedSizeBinary &&
				cmeta->attopts.fixed_size_binary.byteWidth == dtype->byte_width())
				return true;
			__Elog("not compatible FixedSizeBinary column[%d] TYPE=%s byteWidth=%d",
				   kds_col_index,
				   ArrowTypeTagAsCString(cmeta->attopts.tag),
				   cmeta->attopts.fixed_size_binary.byteWidth);
			break;
		case arrow::Type::type::DATE32:
		case arrow::Type::type::DATE64:
			if (cmeta->attopts.tag == ArrowType__Date)
			{
				auto __dtype = std::static_pointer_cast<arrow::DateType>(dtype);
				switch (__dtype->unit())
				{
					case arrow::DateUnit::DAY:
						return (cmeta->attopts.date.unit == ArrowDateUnit__Day);
					case arrow::DateUnit::MILLI:
						return (cmeta->attopts.date.unit == ArrowDateUnit__MilliSecond);
					default:
						break;
				}
				__Elog("not compatible Time column[%d] TYPE=%s unit=%s",
					   kds_col_index,
					   ArrowTypeTagAsCString(cmeta->attopts.tag),
					   ArrowDateUnitAsCString(cmeta->attopts.date.unit));
			}
			else
			{
				__Elog("not compatible Date column[%d] TYPE=%s",
					   kds_col_index,
					   ArrowTypeTagAsCString(cmeta->attopts.tag));
			}
			break;
		case arrow::Type::type::TIMESTAMP:
			if (cmeta->attopts.tag == ArrowType__Timestamp)
			{
				auto __dtype = std::static_pointer_cast<arrow::TimestampType>(dtype);
				switch (__dtype->unit())
				{
					case arrow::TimeUnit::SECOND:
						return (cmeta->attopts.timestamp.unit == ArrowTimeUnit__Second);
					case arrow::TimeUnit::MILLI:
						return (cmeta->attopts.timestamp.unit == ArrowTimeUnit__MilliSecond);
					case arrow::TimeUnit::MICRO:
						return (cmeta->attopts.timestamp.unit == ArrowTimeUnit__MicroSecond);
					case arrow::TimeUnit::NANO:
						return (cmeta->attopts.timestamp.unit == ArrowTimeUnit__NanoSecond);
					default:
						break;
				}
				__Elog("not compatible Timestamp column[%d] TYPE=%s unit=%s",
					   kds_col_index,
					   ArrowTypeTagAsCString(cmeta->attopts.tag),
					   ArrowTimeUnitAsCString(cmeta->attopts.timestamp.unit));
			}
			else
			{
				__Elog("not compatible Timestamp column[%d] TYPE=%s",
					   kds_col_index,
					   ArrowTypeTagAsCString(cmeta->attopts.tag));
			}
			break;
		case arrow::Type::type::TIME32:
		case arrow::Type::type::TIME64:
			if (cmeta->attopts.tag == ArrowType__Time)
			{
				auto __dtype = std::static_pointer_cast<arrow::TimeType>(dtype);
				switch (__dtype->unit())
				{
					case arrow::TimeUnit::SECOND:
						return (cmeta->attopts.time.unit == ArrowTimeUnit__Second);
					case arrow::TimeUnit::MILLI:
						return (cmeta->attopts.time.unit == ArrowTimeUnit__MilliSecond);
					case arrow::TimeUnit::MICRO:
						return (cmeta->attopts.time.unit == ArrowTimeUnit__MicroSecond);
					case arrow::TimeUnit::NANO:
						return (cmeta->attopts.time.unit == ArrowTimeUnit__NanoSecond);
					default:
						break;
				}
				__Elog("not compatible Time column[%d] TYPE=%s unit=%s",
					   kds_col_index,
					   ArrowTypeTagAsCString(cmeta->attopts.tag),
					   ArrowTimeUnitAsCString(cmeta->attopts.time.unit));
			}
			else
			{
				__Elog("not compatible Time column[%d] TYPE=%s",
					   kds_col_index,
					   ArrowTypeTagAsCString(cmeta->attopts.tag));
			}
			break;
		case arrow::Type::type::INTERVAL_MONTHS:
			if (cmeta->attopts.tag == ArrowType__Interval &&
				cmeta->attopts.unitsz == sizeof(int32_t))
				return true;
			__Elog("not compatible Interval column[%d] TYPE=%s unitsz=%d",
				   kds_col_index,
				   ArrowTypeTagAsCString(cmeta->attopts.tag),
				   cmeta->attopts.unitsz);
			break;
		case arrow::Type::type::INTERVAL_DAY_TIME:
			if (cmeta->attopts.tag == ArrowType__Interval &&
				cmeta->attopts.unitsz == sizeof(int64_t))
				return true;
			__Elog("not compatible Interval column[%d] TYPE=%s unitsz=%d",
				   kds_col_index,
				   ArrowTypeTagAsCString(cmeta->attopts.tag),
				   cmeta->attopts.unitsz);
			break;
		case arrow::Type::type::LIST:
		case arrow::Type::type::LARGE_LIST:
			if (cmeta->attopts.tag == ArrowType__List)
			{
				auto __dtype = std::static_pointer_cast<arrow::BaseListType>(dtype);
				if (cmeta->idx_subattrs >= kds_head->ncols &&
					cmeta->idx_subattrs <  kds_head->nr_colmeta &&
					__checkParquetFileColumn(__dtype->value_field(),
											 kds_head, cmeta->idx_subattrs))
					return true;
				__Elog("List subfield is out of range");
			}
			else
			{
				__Elog("not compatible List column[%d] TYPE=%s",
					   kds_col_index,
					   ArrowTypeTagAsCString(cmeta->attopts.tag));
			}
			break;
		case arrow::Type::type::STRUCT:
			if (cmeta->attopts.tag == ArrowType__Struct)
			{
				auto __dtype = std::static_pointer_cast<arrow::StructType>(dtype);
				if (__dtype->num_fields() == cmeta->num_subattrs &&
					cmeta->idx_subattrs >= kds_head->ncols &&
					cmeta->idx_subattrs +
					cmeta->num_subattrs <= kds_head->nr_colmeta)
				{
					for (int k=0; k < __dtype->num_fields(); k++)
					{
						if (!__checkParquetFileColumn(__dtype->field(k),
													  kds_head, cmeta->idx_subattrs + k))
							return false;
					}
					return true;
				}
				__Elog("Struct subfield is out of range");
			}
			else
			{
				__Elog("not compatible Struct column[%d] TYPE=%s",
					   kds_col_index,
					   ArrowTypeTagAsCString(cmeta->attopts.tag));
			}
			break;
		default:
			__Elog("not compatible Unknown column[%d] TYPE=%s",
				   kds_col_index,
				   ArrowTypeTagAsCString(cmeta->attopts.tag));
			break;
	}
	return false;
}

/*
 * checkParquetFileSchema
 */
static bool
checkParquetFileSchema(std::shared_ptr<arrow::Schema> arrow_schema,
					   const kern_data_store *kds_head)
{
	// Type compatibility checks (only referenced attributes)
	for (int j=0; j < kds_head->ncols; j++)
	{
		int		field_index = kds_head->colmeta[j].field_index;

		if (field_index < 0)
			continue;
		if (field_index < arrow_schema->num_fields())
		{
			auto	field = arrow_schema->field(field_index);

			if (!__checkParquetFileColumn(field, kds_head, j))
			{
				__Elog("field_index = %d not compatible", field_index);
				return false;	/* not compatible */
			}
		}
		else
		{
			__Elog("not compatible Schema: field index %d out of range [%d]",
				   field_index, arrow_schema->num_fields());
			return false;	/* out of range*/
		}
	}
	return true;
}

/*
 * parquetReadArrowTable
 */
static kern_data_store *
parquetReadArrowTable(std::shared_ptr<arrow::Table> table,
					  std::vector<int> referenced,
					  const kern_data_store *kds_head,
					  void *(*malloc_callback)(void *malloc_private,
											   size_t malloc_size),
					  void *malloc_private)
{
	size_t		kds_length = KDS_HEAD_LENGTH(kds_head) + kds_head->arrow_virtual_usage;
	size_t		curr_pos = kds_length;
	kern_data_store *kds;

	/*
	 * estimate the buffer length
	 */
	assert(kds_head->format == KDS_FORMAT_PARQUET);
	for (int k=0; k < table->num_columns(); k++)
	{
		auto	column = table->column(k);
		for (const auto &chunk : column->chunks())
		{
			auto	data = chunk->data();
			int		count = 0;
			for (const auto &buf : data->buffers)
			{
				if (buf)
					kds_length += ARROW_ALIGN(buf->size());
				if (++count > 3)
				{
					__Elog("unknown buffer layout");
					return NULL;	/* unknown buffer layout */
				}
			}
		}
	}
	/*
	 * buffer allocation
	 */
	kds = (kern_data_store *)malloc_callback(malloc_private, kds_length);
	if (!kds)
	{
		__Elog("out of memory");
		return NULL;
	}
	/*
	 * fillup the buffer
	 */
	memcpy(kds, kds_head, curr_pos);
	for (int k=0; k < table->num_columns(); k++)
	{
		auto	column = table->column(k);
		auto	cmeta = &kds->colmeta[referenced[k]];

		for (const auto &chunk : column->chunks())
		{
			auto	data = chunk->data();
			int		phase = 0;
			for (const auto &buf : data->buffers)
			{
				uint64_t	__offset = 0;
				uint64_t	__length = 0;

				if (buf)
				{
					__offset = curr_pos;
					__length = buf->size();
					memcpy((char *)kds + __offset, buf->data(), buf->size());
					curr_pos += ARROW_ALIGN(__length);
				}
				switch (phase)
				{
					case 0:
						cmeta->nullmap_offset = __offset;
						cmeta->nullmap_length = __length;
						break;
					case 1:
						cmeta->values_offset = __offset;
						cmeta->values_length = __length;
						break;
					default:
						assert(phase == 2);
						cmeta->extra_offset = __offset;
						cmeta->extra_length = __length;
						break;
				}
				phase++;
			}
		}
	}
	kds->format = KDS_FORMAT_ARROW;

	return kds;
}

/*
 * parquetReadOneRowGroup
 *
 * It returns a KDS buffer with KDS_FORMAT_ARROW that loads the
 * specified row-group.
 */
static kern_data_store *
__parquetReadOneRowGroup(const char *filename,
						 const kern_data_store *kds_head,
						 void *(*malloc_callback)(void *malloc_private,
												  size_t malloc_size),
						 void *malloc_private,
						 uint32_t *p_nrowgroups_cudf_read)
{
	uint32_t	row_group_index = kds_head->parquet_row_group;
	struct stat	stat_buf;
	uint32_t	hash, hindex;
	parquetMetaDataCache *entry;
	std::shared_ptr<parquet::FileMetaData> metadata = nullptr;
    std::unique_ptr<parquet::ParquetFileReader> raw_file_reader = nullptr;
    std::unique_ptr<parquet::arrow::FileReader> parquet_file_reader = nullptr;
	std::shared_ptr<arrow::Schema> arrow_schema;
	kern_data_store *kds = NULL;
	arrow::Status status;

	/*
	 * Open a new Parquet File Reader dedicated for this thread, but
	 * utilize parquet::FileMetada once parsed by other one.
	 * As discussed in #937, libarrow/libparquet is not designed for
	 * thread-safe, and caller must take care mutual controls.
	 */

	// Lookup the local file metadata cache first
	if (stat(filename, &stat_buf) != 0)
	{
		__Elog("failed on stat('%s'): %m", filename);
		return NULL;
	}
	hash = __parquetLocalFileHash(stat_buf.st_dev,
								  stat_buf.st_ino);
	hindex = hash % PQ_HASH_NSLOTS;

	pq_hash_lock[hindex].lock();
	__dlist_foreach(entry, &pq_hash_slot[hindex])
	{
		if (entry->stat_buf.st_dev == stat_buf.st_dev &&
			entry->stat_buf.st_ino == stat_buf.st_ino)
		{
			// confirm the stat_buf.st_mtim is identical to the hashed one.
			// if changed, it means the parquet file is modified on the storage.
			if (entry->stat_buf.st_mtim.tv_sec  == stat_buf.st_mtim.tv_sec &&
				entry->stat_buf.st_mtim.tv_nsec == stat_buf.st_mtim.tv_nsec)
			{
				entry->refcnt++;
				metadata = entry->metadata;
				pq_hash_lock[hindex].unlock();
				break;
			}
			// Oops, the parquet file was modified on the disk, so should be
			// invalid by the one who hold this entry.
			assert(entry->refcnt > 0);
		}
	}
	/*
	 * Open the Parquet File (with cached metadata)
	 */
	raw_file_reader = parquet::ParquetFileReader::OpenFile(std::string(filename),
														   false,	/* memory_map */
														   parquet::default_reader_properties(),
														   metadata);
	if (!raw_file_reader)
	{
		if (entry)
			parquetPutMetaDataCache(entry);
		else
			pq_hash_lock[hindex].unlock();
		__Elog("failed on parquet::ParquetFileReader::Open('%s')", filename);
		return NULL;
	}

	/*
	 * Add metadata to the metadata cache (if not cached yet)
	 */
	if (!entry)
	{
		assert(!metadata);
		entry = new(std::nothrow) parquetMetaDataCache(filename, &stat_buf, hash);
		if (!entry)
		{
			pq_hash_lock[hindex].unlock();
			__Elog("out of memory");
			return NULL;
		}
		entry->metadata = raw_file_reader->metadata();
		pq_hash_lock[hindex].unlock();
	}
	/*
	 * Open the Arrow File Reader
	 */
	status = parquet::arrow::FileReader::Make(arrow::default_memory_pool(),
											  std::move(raw_file_reader),
											  &parquet_file_reader);
	if (!status.ok())
	{
		parquetPutMetaDataCache(entry);
		__Elog("failed on parquet::arrow::FileReader::Make('%s'): %s",
			   filename, status.ToString().c_str());
		return NULL;
	}
	// quick check of schema compatibility
	status = parquet_file_reader->GetSchema(&arrow_schema);
	if (!status.ok())
	{
		parquetPutMetaDataCache(entry);
		__Elog("failed on parquet::arrow::FileReader::GetSchema(): %s",
			   status.ToString().c_str());
		return NULL;
	}
	if (checkParquetFileSchema(arrow_schema, kds_head) &&
		row_group_index < parquet_file_reader->num_row_groups())
	{
		std::shared_ptr<arrow::Table> table;
		std::vector<int>	referenced;

		for (int j=0; j < kds_head->ncols; j++)
		{
			auto	cmeta = &kds_head->colmeta[j];

			if (cmeta->field_index >= 0)
				referenced.push_back(cmeta->field_index);
		}

		/* Try cuDF GPU-native Parquet reading if enabled */
		bool gpu_reading_successful = false;
		if (arrow_fdw_gpu_decompression_enabled && !referenced.empty())
		{
			fprintf(stderr, "[PG-Strom] Attempting cuDF GPU-native Parquet reading for row group %u\n",
					row_group_index);
			fflush(stderr);

			const char *cudf_error = nullptr;
			kds = parquetReadRowGroupCuDF(filename, row_group_index,
										  referenced, kds_head,
										  malloc_callback, malloc_private,
										  &cudf_error);
			if (kds)
			{
				fprintf(stderr, "[PG-Strom] cuDF GPU-native reading completed for row group %u\n", row_group_index);
				fflush(stderr);
				gpu_reading_successful = true;

				/* Increment cuDF usage counter if provided */
				if (p_nrowgroups_cudf_read)
					(*p_nrowgroups_cudf_read)++;
			}
			else
			{
				fprintf(stderr, "[PG-Strom] cuDF failed: %s, falling back to CPU\n",
						cudf_error ? cudf_error : "unknown");
				fflush(stderr);
			}
		}

		/* Fall back to CPU decompression (standard Arrow reader) if cuDF failed */
		if (!gpu_reading_successful)
		{
			if (!arrow_fdw_gpu_decompression_enabled)
			{
				fprintf(stderr, "INFO: Using CPU decompression (GPU reading disabled)\n");
			}

			status = parquet_file_reader->ReadRowGroup(row_group_index,
													   referenced,
													   &table);
			if (status.ok())
			{
				kds = parquetReadArrowTable(table, referenced,
											kds_head,
											malloc_callback,
											malloc_private);
			}
			else
			{
				__Elog("failed on parquet::arrow::FileReader::ReadRowGroup: %s",
					   status.ToString().c_str());
			}
		}
	}
	parquetPutMetaDataCache(entry);
	return kds;
}

/*
 * parquetReadOneRowGroup - interface to C-portion
 */
kern_data_store *
parquetReadOneRowGroup(const char *filename,
					   const kern_data_store *kds_head,
					   void *(*malloc_callback)(void *malloc_private,
												size_t malloc_size),
					   void *malloc_private,
					   uint32_t *p_nrowgroups_cudf_read,
					   const char **p_error_message)
{
	kern_data_store *kds = NULL;

	*__private_error_message = '\0';
	try {
		kds = __parquetReadOneRowGroup(filename,
									   kds_head,
									   malloc_callback,
									   malloc_private,
									   p_nrowgroups_cudf_read);
	}
	catch (const std::exception &e) {
		const char *estr = e.what();
		snprintf(__private_error_message,
				 sizeof(__private_error_message),
				 "[exception] %s", estr);
	}
	/* error reporting */
	if (p_error_message)
	{
		if (kds)
			*p_error_message = NULL;	/* no error status */
		else if (*__private_error_message == '\0')
			*p_error_message = "unknown internal error";
		else
			*p_error_message = __private_error_message;
	}
	return kds;
}

/*
 * ParquetColumnStats
 *
 * Structure to hold aggregated statistics from Parquet metadata
 * across all row groups in all files for a single column.
 */
struct ParquetColumnStats
{
	int64_t		total_rows;			/* Total number of rows across all files/row-groups */
	int64_t		null_count;			/* Total number of NULL values */
	bool		has_min_max;		/* True if min/max are available */
	parquet::Type::type physical_type;	/* Parquet physical type */

	/* Serialized min/max values (size depends on type) */
	std::vector<uint8_t> min_value;
	std::vector<uint8_t> max_value;

	/* Average width estimate for variable-length types */
	int32_t		avg_width;

	ParquetColumnStats()
		: total_rows(0), null_count(0), has_min_max(false),
		  physical_type(parquet::Type::BOOLEAN), avg_width(0)
	{
	}
};

/*
 * __gatherColumnStatistics
 *
 * Helper function to extract statistics for a single column from
 * a single Parquet file's metadata, accumulating into the stats object.
 */
static bool
__gatherColumnStatistics(const std::shared_ptr<parquet::FileMetaData> &file_metadata,
						 int column_index,
						 ParquetColumnStats *stats)
{
	int num_row_groups = file_metadata->num_row_groups();

	for (int rg_idx = 0; rg_idx < num_row_groups; rg_idx++)
	{
		auto rg_metadata = file_metadata->RowGroup(rg_idx);

		/* Validate column index */
		if (column_index >= rg_metadata->num_columns())
		{
			__Elog("column index %d out of range (max %d)",
				   column_index, rg_metadata->num_columns());
			return false;
		}

		auto col_chunk = rg_metadata->ColumnChunk(column_index);
		stats->total_rows += rg_metadata->num_rows();

		/* Check if statistics are available */
		if (!col_chunk->is_stats_set())
			continue;

		auto col_stats = col_chunk->statistics();
		if (!col_stats)
			continue;

		/* Accumulate null count */
		stats->null_count += col_stats->null_count();

		/* Get physical type (should be same across all row groups) */
		/* Note: Physical type is available from schema descriptor, not ColumnChunkMetaData */
		/* For now, we'll determine it from the statistics type later */

		/* Accumulate min/max if available */
		if (col_stats->HasMinMax())
		{
			if (!stats->has_min_max)
			{
				/* First row group with min/max - initialize */
				stats->has_min_max = true;
				stats->min_value = std::vector<uint8_t>(
					col_stats->EncodeMin().begin(),
					col_stats->EncodeMin().end());
				stats->max_value = std::vector<uint8_t>(
					col_stats->EncodeMax().begin(),
					col_stats->EncodeMax().end());
			}
			else
			{
				/* Update min/max across row groups */
				std::string encoded_min = col_stats->EncodeMin();
				std::string encoded_max = col_stats->EncodeMax();

				/* Compare and update min */
				if (memcmp(encoded_min.data(), stats->min_value.data(),
						   std::min(encoded_min.size(), stats->min_value.size())) < 0)
				{
					stats->min_value = std::vector<uint8_t>(
						encoded_min.begin(), encoded_min.end());
				}

				/* Compare and update max */
				if (memcmp(encoded_max.data(), stats->max_value.data(),
						   std::min(encoded_max.size(), stats->max_value.size())) > 0)
				{
					stats->max_value = std::vector<uint8_t>(
						encoded_max.begin(), encoded_max.end());
				}
			}
		}
	}

	return true;
}

/*
 * __parquetGatherMetadataStatistics
 *
 * Internal C++ implementation to gather statistics from Parquet metadata.
 * Iterates through all files and all row groups to aggregate statistics
 * for a single column without decompressing any data.
 */
static ParquetColumnStats *
__parquetGatherMetadataStatistics(const char **filenames,
								  int num_files,
								  int column_index)
{
	ParquetColumnStats *stats = new(std::nothrow) ParquetColumnStats();
	if (!stats)
	{
		__Elog("out of memory");
		return NULL;
	}

	for (int file_idx = 0; file_idx < num_files; file_idx++)
	{
		const char *filename = filenames[file_idx];
		struct stat stat_buf;
		uint32_t hash, hindex;
		parquetMetaDataCache *entry = nullptr;
		std::shared_ptr<parquet::FileMetaData> metadata = nullptr;

		/* Stat the file */
		if (stat(filename, &stat_buf) != 0)
		{
			__Elog("failed on stat('%s'): %m", filename);
			delete stats;
			return NULL;
		}

		hash = __parquetLocalFileHash(stat_buf.st_dev, stat_buf.st_ino);
		hindex = hash % PQ_HASH_NSLOTS;

		/* Try to get from cache */
		pq_hash_lock[hindex].lock();
		__dlist_foreach(entry, &pq_hash_slot[hindex])
		{
			if (entry->stat_buf.st_dev == stat_buf.st_dev &&
				entry->stat_buf.st_ino == stat_buf.st_ino)
			{
				if (entry->stat_buf.st_mtim.tv_sec == stat_buf.st_mtim.tv_sec &&
					entry->stat_buf.st_mtim.tv_nsec == stat_buf.st_mtim.tv_nsec)
				{
					entry->refcnt++;
					metadata = entry->metadata;
					pq_hash_lock[hindex].unlock();
					break;
				}
			}
		}

		/* Open file if not in cache */
		if (!metadata)
		{
			pq_hash_lock[hindex].unlock();

			try {
				auto raw_reader = parquet::ParquetFileReader::OpenFile(
					std::string(filename),
					false,	/* memory_map */
					parquet::default_reader_properties(),
					nullptr);

				if (!raw_reader)
				{
					__Elog("failed to open Parquet file '%s'", filename);
					delete stats;
					return NULL;
				}

				metadata = raw_reader->metadata();

				/* Add to cache */
				pq_hash_lock[hindex].lock();
				entry = new(std::nothrow) parquetMetaDataCache(filename, &stat_buf, hash);
				if (entry)
				{
					entry->metadata = metadata;
					__dlist_push_tail(&pq_hash_slot[hindex], &entry->chain);
				}
				pq_hash_lock[hindex].unlock();
			}
			catch (const std::exception &e) {
				__Elog("exception opening Parquet file '%s': %s", filename, e.what());
				delete stats;
				return NULL;
			}
		}

		/* Gather statistics from this file */
		if (!__gatherColumnStatistics(metadata, column_index, stats))
		{
			if (entry)
				parquetPutMetaDataCache(entry);
			delete stats;
			return NULL;
		}

		if (entry)
			parquetPutMetaDataCache(entry);
	}

	/* Estimate average width for variable-length types */
	if (stats->physical_type == parquet::Type::BYTE_ARRAY ||
		stats->physical_type == parquet::Type::FIXED_LEN_BYTE_ARRAY)
	{
		if (stats->has_min_max && stats->min_value.size() > 0 && stats->max_value.size() > 0)
		{
			/* Simple estimate: average of min and max lengths */
			stats->avg_width = (stats->min_value.size() + stats->max_value.size()) / 2;
		}
		else
		{
			/* Default estimate for strings */
			stats->avg_width = 32;
		}
	}
	else
	{
		/* Fixed-size types */
		switch (stats->physical_type)
		{
			case parquet::Type::BOOLEAN:
				stats->avg_width = 1;
				break;
			case parquet::Type::INT32:
			case parquet::Type::FLOAT:
				stats->avg_width = 4;
				break;
			case parquet::Type::INT64:
			case parquet::Type::DOUBLE:
				stats->avg_width = 8;
				break;
			case parquet::Type::INT96:
				stats->avg_width = 12;
				break;
			default:
				stats->avg_width = 8;
				break;
		}
	}

	return stats;
}

/*
 * parquetGatherMetadataStatistics - C interface
 *
 * Gathers statistics for a single column from Parquet file metadata
 * across multiple files without decompressing data.
 *
 * Returns a ParquetColumnStats structure or NULL on error.
 * Caller must call parquetFreeColumnStats() to free the result.
 */
extern "C" void *
parquetGatherMetadataStatistics(const char **filenames,
							   int num_files,
							   int column_index,
							   const char **p_error_message)
{
	ParquetColumnStats *stats = NULL;

	*__private_error_message = '\0';
	try {
		stats = __parquetGatherMetadataStatistics(filenames,
												  num_files,
												  column_index);
	}
	catch (const std::exception &e) {
		snprintf(__private_error_message,
				 sizeof(__private_error_message),
				 "[exception] %s", e.what());
	}

	/* error reporting */
	if (p_error_message)
	{
		if (stats)
			*p_error_message = NULL;	/* no error status */
		else if (*__private_error_message == '\0')
			*p_error_message = "unknown internal error";
		else
			*p_error_message = __private_error_message;
	}

	return (void *)stats;
}

/*
 * parquetFreeColumnStats - C interface
 *
 * Frees a ParquetColumnStats structure allocated by
 * parquetGatherMetadataStatistics().
 */
extern "C" void
parquetFreeColumnStats(void *stats_ptr)
{
	if (stats_ptr)
		delete ((ParquetColumnStats *)stats_ptr);
}

/*
 * Accessor functions for ParquetColumnStats (C interface)
 * These allow C code to extract fields from the C++ structure.
 */
extern "C" int64_t
parquetColumnStatsTotalRows(void *stats_ptr)
{
	if (!stats_ptr)
		return 0;
	return ((ParquetColumnStats *)stats_ptr)->total_rows;
}

extern "C" int64_t
parquetColumnStatsNullCount(void *stats_ptr)
{
	if (!stats_ptr)
		return 0;
	return ((ParquetColumnStats *)stats_ptr)->null_count;
}

extern "C" bool
parquetColumnStatsHasMinMax(void *stats_ptr)
{
	if (!stats_ptr)
		return false;
	return ((ParquetColumnStats *)stats_ptr)->has_min_max;
}

extern "C" int
parquetColumnStatsPhysicalType(void *stats_ptr)
{
	if (!stats_ptr)
		return 0;
	return (int)((ParquetColumnStats *)stats_ptr)->physical_type;
}

extern "C" const uint8_t *
parquetColumnStatsMinValue(void *stats_ptr, size_t *p_size)
{
	if (!stats_ptr)
	{
		*p_size = 0;
		return NULL;
	}
	auto stats = (ParquetColumnStats *)stats_ptr;
	*p_size = stats->min_value.size();
	return stats->min_value.data();
}

extern "C" const uint8_t *
parquetColumnStatsMaxValue(void *stats_ptr, size_t *p_size)
{
	if (!stats_ptr)
	{
		*p_size = 0;
		return NULL;
	}
	auto stats = (ParquetColumnStats *)stats_ptr;
	*p_size = stats->max_value.size();
	return stats->max_value.data();
}

extern "C" int32_t
parquetColumnStatsAvgWidth(void *stats_ptr)
{
	if (!stats_ptr)
		return 0;
	return ((ParquetColumnStats *)stats_ptr)->avg_width;
}
