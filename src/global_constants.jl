default_block_size() = Bool(VectorizationBase.has_feature(Val(:x86_64_avx512f))) ? 96 : 64

