# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Trần Nguyễn Anh Thư
**Nhóm:** 069
**Ngày:** 05/06/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Khi hai chunk có high cosine similarity nghĩa là đang có độ tương đồng về ngữ nghĩa bất kể độ dài, ngữ cảnh hay cách hành văn khác nhau. 

**Ví dụ HIGH similarity:**
- Sentence A: Mô hình nhận diện hình ảnh dự đoán sai lệch do tập dữ liệu huấn luyện bị thiếu hụt
- Sentence B: Object detection model trả về kết quả kém vì dataset đầu vào không đủ độ đa dạng
- Tại sao tương đồng: Cả 2 đều có thể được rút gọn lại thành một ý chính

**Ví dụ LOW similarity:**
- Sentence A: Mô hình AI bị lỗi
- Sentence B: Tôi hát rất hay
- Tại sao khác: Không liên quan

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Vì cosine similarity không quan tâm đến độ dài của vector, chỉ quan tâm đến hướng của vector. Còn Euclidean lại bị ảnh hưởng bởi độ dài của vector.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Số chunks = (10000 - 50) / (500 - 50) = 23 chunks.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Overlap tăng lên 100 thì chunk count cũng tăng lên ~ 25 chunks. Việc cắt text cứng nhắc theo đúng số lượng ký tự là cực kỳ rủi ro. Tăng overlap sẽ giúp giải quyết hai bài toán cốt lõi là Bảo toàn ngữ cảnh và Tối ưu khả năng truy xuất.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Điều khoản dịch vụ

**Tại sao nhóm chọn domain này?**
> Nhóm chọn domain 'Điều khoản dịch vụ' vì đặc thù văn bản có tính cấu trúc chặt chẽ, ngôn từ pháp lý phức tạp và dễ bị mất ngữ cảnh nếu chunking sai. Tập dữ liệu này phù hợp để đánh giá độ chính xác của các thuật toán phân tách cấu trúc, đồng thời cung cấp các thuộc tính rõ ràng để thử nghiệm khả năng truy xuất kết hợp Metadata Filtering

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | `be_tos.txt` | https://be.com.vn/ho-tro/dieu-khoan-su-dung/ | 3,541 | source=be, type=tos, language=vi |
| 2 | `grab_tos.txt` | https://www.grab.com/vn/terms-policies/transport-delivery-logistics/ | 223,216 | source=grab, type=tos, language=vi |
| 3 | `shopee_tos.txt` | https://help.shopee.vn/portal/4/article/77243?previousPage=other+articles | 82,685 | source=shopee, type=tos, language=vi |
| 4 | `viettelpost_tos.txt` | https://viettelpost.com.vn/guide/dieu-khoan-va-quy-dinh/ | 14,361 | source=viettelpost, type=tos, language=vi |
| 5 | `zalopay_tos.txt` | https://zalopay.vn/dich-vu/dieu-khoan-su-dung | 21,726 | source=zalopay, type=tos, language=vi |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| source | string | `grab`, `shopee`, `viettelpost` | Filter theo nền tảng khi query hỏi về một công ty cụ thể — tránh retrieve chunk từ ToS của Shopee khi hỏi về ViettelPost |
| type | string | `tos` | Phân biệt với file dạng khác nếu mở rộng thêm loại tài liệu (e.g. FAQ, policy) |
| language | string | `vi` | Hữu ích khi tập dữ liệu đa ngôn ngữ; có thể filter language=vi để tránh lẫn chunk tiếng Anh |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| `be_tos.txt` | FixedSizeChunker (`fixed_size`) | 20 | 196.05 | Thấp. Cắt ngang câu/từ. |
| `be_tos.txt` | SentenceChunker (`by_sentences`) | 6 | 589.33 | Trung bình. Giữ nguyên câu nhưng gom quá nhiều câu thành 1 chunk. |
| `be_tos.txt` | RecursiveChunker (`recursive`) | 23 | 153.96 | Tốt. Cắt theo đoạn văn (`\n\n`) hoặc câu (`. `), giúp giữ trọn vẹn một ý. |
| `viettelpost_tos.txt` | FixedSizeChunker (`fixed_size`) | 80 | 199.26 | Thấp. Dễ chia cắt các quy định pháp lý. |
| `viettelpost_tos.txt` | SentenceChunker (`by_sentences`) | 40 | 358.05 | Khá. Tốt hơn Fixed Size nhưng dễ bị sót các cấu trúc gạch đầu dòng. |
| `viettelpost_tos.txt` | RecursiveChunker (`recursive`) | 88 | 163.19 | Rất Tốt. Tôn trọng các đoạn ngắt dòng của các điều khoản. |
| `zalopay_tos.txt` | FixedSizeChunker (`fixed_size`) | 121 | 199.39 | Thấp. Tearing (rách ý) xảy ra ở giữa các câu khoản dài. |
| `zalopay_tos.txt` | SentenceChunker (`by_sentences`) | 43 | 502.72 | Trung bình. Sót các định dạng dòng liệt kê ngắn. |
| `zalopay_tos.txt` | RecursiveChunker (`recursive`) | 144 | 150.88 | Rất Tốt. Giữ được các khoản mục và danh sách liệt kê trọn vẹn. |

### Strategy Của Tôi

**Loại:** Sliding Window with Overlap (`FixedSizeChunker` với overlap)

**Mô tả cách hoạt động:**
> Chia text thành các chunk kích thước cố định (`chunk_size=300` ký tự), mỗi chunk overlap với chunk trước `overlap=50` ký tự. Window trượt từ đầu đến cuối với bước `step = chunk_size - overlap = 250`. Nhờ phần overlap, nội dung tại biên giới giữa hai chunk liền kề luôn xuất hiện đầy đủ trong ít nhất một trong hai chunk — giảm nguy cơ mất ngữ cảnh khi câu bị cắt ngang.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Văn bản điều khoản dịch vụ thường có các câu khoản dài, không phải lúc nào cũng có cấu trúc "Điều X" rõ ràng. Sliding window đảm bảo mọi đoạn văn đều được bao phủ với độ dài đồng đều, giúp retrieval ổn định hơn; phần overlap đặc biệt hữu ích khi câu khoản quan trọng nằm đúng ở biên giới giữa hai chunk liên tiếp.

**Code snippet (nếu custom):**
```python
class SlidingWindowChunker:
    def __init__(self, chunk_size: int = 300, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        step = self.chunk_size - self.overlap
        chunks = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| `viettelpost_tos.txt` | RecursiveChunker (Best Baseline) | 88 | 163.19 | Tốt, tôn trọng cấu trúc đoạn văn, nhưng chunk count rất lớn làm tốn thời gian search. |
| `viettelpost_tos.txt` | **Sliding Window (Của tôi)** | 58 | 296.7 | Khá tốt. Chunk đồng đều, overlap 50 ký tự đảm bảo câu khoản tại biên không bị bỏ sót. Số chunk ít hơn nhiều so với RecursiveChunker (58 vs 88), dễ kiểm soát độ lớn context. |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Anh Thư - 2A202600915 | Sliding Window with Overlap (chunk_size=300, overlap=50) | 2/10 | Chunk đồng đều, overlap bảo toàn ngữ cảnh tại biên, số lượng chunk ít và kiểm soát được | Cắt ngang câu/từ, không tôn trọng cấu trúc Điều khoản, chunk trùng lặp tăng storage |
| Hữu Khoa - 2A202600863 | RecursiveChunker | 2/10 | Chunk nhỏ gọn, bao phủ nhiều đoạn hơn | Không biết một chunk đang thuộc Điều nào |
| Duy Bảo - 2A202600688 | SentenceChunker | 2/10 | Giữ trọn câu, dễ implement | Gộp quá nhiều câu không liên quan vào 1 chunk |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> RecursiveChunker vẫn là phương án baseline tốt nhất vì tôn trọng cấu trúc tự nhiên của văn bản pháp lý (đoạn, câu) thay vì cắt cứng theo số ký tự. Sliding Window with Overlap là lựa chọn thực tế khi cần chunk count nhỏ và dự đoán được — overlap giải quyết được bài toán context boundary mà FixedSize thuần không làm được. Nếu văn bản có cấu trúc Điều/Khoản rõ ràng, một custom legal chunker sẽ cho retrieval tốt nhất nhưng phức tạp hơn.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Dùng regex `([.!?] |\.\n)` để nhận diện cuối câu, scan qua text theo từng part — mỗi khi gặp dấu câu thì flush `current` buffer vào danh sách `sentences`. Sau đó gom từng nhóm `max_sentences_per_chunk` câu thành 1 chunk bằng `" ".join(...)`. Edge case: đoạn text cuối không có dấu câu vẫn được thu thập nhờ kiểm tra `current.strip()` sau vòng lặp.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Base case: nếu `len(current_text) <= chunk_size` thì trả về ngay `[current_text]`. Ngược lại, thử separator đầu tiên trong danh sách ưu tiên `["\n\n", "\n", ". ", " ", ""]`; nếu separator không xuất hiện trong text thì đệ quy với separator tiếp theo. Sau khi split, áp dụng greedy bin-packing: cộng dồn parts vào `current_chunk` cho đến khi vượt `chunk_size` thì flush — tránh sinh ra quá nhiều chunk nhỏ vụn.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> `add_documents` gọi `_embedding_fn(doc.content)` cho từng document rồi lưu dict `{id, content, metadata, embedding}` vào `self._store`. `search` embed query, tính dot product giữa query embedding với toàn bộ stored embeddings (do vector đã được normalize L2, dot product tương đương cosine similarity), sort descending và slice `[:top_k]`.

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter` **filter trước**: lọc `self._store` giữ lại records có tất cả key-value trong `metadata_filter` khớp, sau đó mới chạy similarity search trên tập đã thu hẹp — hiệu quả hơn vì tránh tính embedding cho chunk không liên quan. `delete_document` dùng list comprehension loại bỏ mọi record có `metadata["doc_id"] == doc_id` và trả `True` nếu `len(self._store)` giảm.

### KnowledgeBaseAgent

**`answer`** — approach:
> Gọi `store.search(question, top_k)` lấy top-k chunks, join `content` bằng `\n` thành chuỗi `context`. Inject vào prompt template `"Context:\n{context}\n\nQuestion: {question}\nAnswer:"` — cấu trúc này đặt context trước câu hỏi giúp LLM ưu tiên thông tin đã cung cấp trước khi dùng kiến thức nội bộ. Toàn bộ prompt được truyền cho `llm_fn` và kết quả trả về trực tiếp.

### Test Results

```
================================================ test session starts ================================================
platform darwin -- Python 3.13.9, pytest-8.4.2, pluggy-1.5.0 -- /opt/anaconda3/bin/python
cachedir: .pytest_cache
rootdir: /Users/realzoey/Desktop/2A202600915-TranNguyenAnhThu-Day07
plugins: anyio-4.10.0
collected 42 items                                                                                                  

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED                         [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED                                  [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED                           [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED                            [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED                                 [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED                 [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED                       [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED                        [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED                      [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED                                        [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED                        [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED                                   [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED                               [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED                                         [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED                [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED                    [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED              [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED                    [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED                                        [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED                          [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED                            [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED                                  [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED                       [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED                         [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED             [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED                          [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED                                   [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED                                  [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED                             [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED                         [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED                    [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED                        [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED                              [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED                        [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED     [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED                   [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED                  [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED      [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED                 [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED          [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED    [100%]

================================================ 42 passed in 0.13s =================================================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Users may cancel their account at any time. | You can delete your account whenever you want. | high | -0.0436 | ✗ |
| 2 | Grab is not liable for any indirect damages. | The weather is nice today, I went for a walk in the park. | low | -0.0910 | ✓ |
| 3 | Shopee may suspend accounts that violate the terms of service. | ViettelPost may deactivate accounts if users breach regulations. | high | -0.0102 | ✗ |
| 4 | Users must provide accurate information when registering. | Orders will be delivered within 3 to 5 business days. | low | -0.0319 | ✓ |
| 5 | ZaloPay secures users payment information. | The e-wallet app encrypts transaction data to protect users. | high | -0.0992 | ✗ |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Bất ngờ nhất là Pair 1 và 3 — hai câu có nghĩa gần như đồng nhất nhưng MockEmbedder vẫn cho điểm âm gần 0, tức là coi chúng "ngược chiều". MockEmbedder dùng MD5 hash để sinh vector ngẫu nhiên xác định, hoàn toàn không học ngữ nghĩa, nên scores chỉ phản ánh đặc điểm chuỗi ký tự chứ không phản ánh meaning. Điều này lý giải tại sao Section 6 chỉ đạt 3/5 trong top-3 dù đã đổi sang tiếng Anh — cải thiện chủ yếu nhờ corpus cân bằng hơn, không phải nhờ embedding hiểu nghĩa tốt hơn.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | What is the minimum age required to use Grab services and sign the agreement? | 18 years old. Users under 18 cannot enter into this Agreement. |
| 2 | How long does ViettelPost take to resolve a complaint? | No more than 2 months for domestic services; 3 months for international services. |
| 3 | What is the time limit to file a complaint about a lost ViettelPost package? | 6 months from the end of the delivery period. |
| 4 | For what reasons can Shopee delete a user account? | Inactive account, violation of terms, fraud/harassment, multiple accounts, coupon abuse, or false information. |
| 5 | What should a user do if they disagree with Grab's updated terms? | Stop using the service and delete the app. Continued use implies acceptance of the new terms. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | What is the minimum age required to use Grab services and sign the agreement? | ZaloPay — đoạn về quy trình thông tin giao dịch (sai) | 0.3393 | No | Không trả lời được — context không liên quan |
| 2 | How long does ViettelPost take to resolve a complaint? | Shopee — đoạn về password và tài khoản (sai top-1, ViettelPost có ở top-2, top-3) | 0.2688 | No (top-3: Yes) | Không trả lời được |
| 3 | What is the time limit to file a complaint about a lost ViettelPost package? | **ViettelPost — đoạn về thời hạn khiếu nại bưu gửi** | 0.2320 | **Yes** | 1 tháng kể từ ngày giao cho khiếu nại hàng hư hỏng |
| 4 | For what reasons can Shopee delete a user account? | Be — đoạn về thay đổi điều khoản (sai) | 0.3425 | No | Không trả lời được |
| 5 | What should a user do if they disagree with Grab updated terms? | Be — đoạn về giải quyết tranh chấp (sai top-1, Grab có ở top-2) | 0.2769 | No (top-3: Yes) | Không trả lời được |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 3 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Tôi học được cách RecursiveChunker hoạt động theo nguyên tắc "separator priority" — thay vì cắt cứng theo ký tự, nó tôn trọng cấu trúc tự nhiên của văn bản từ thô đến chi tiết (đoạn → câu → từ). Cách tiếp cận này linh hoạt hơn nhiều so với Legal Section Chunker của tôi khi gặp văn bản không có Điều/Khoản rõ ràng.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Một nhóm demo việc kết hợp metadata filtering với hybrid search (keyword + vector) — thay vì chỉ dùng cosine similarity, họ đánh trọng số thêm BM25 keyword score để tăng precision khi query chứa từ khóa chuyên ngành xuất hiện chính xác trong document. Đây là cải tiến thực tế quan trọng mà pure vector search không làm được.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Trước tiên sẽ dùng embedding model thực thay vì MockEmbedder vì kết quả query quá tệ, 0/5 queries retrieve đúng. Ngoài ra, sẽ bổ sung thêm metadata `section` (tên Điều cụ thể) để hỗ trợ metadata filtering theo điều khoản thay vì chỉ filter theo `source`.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **100 / 100** |
