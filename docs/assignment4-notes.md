[TOC]

# 2 Filtering Common Crawl

## 2.1 Looking at the data

**问题：look_at_cc**

题目给出的 WARC 和 WET 文件已下载到：

```text
assignment4-data/data/CC-MAIN-20250417135010-20250417165010-00065.warc.gz
assignment4-data/data/CC-MAIN-20250417135010-20250417165010-00065.warc.wet.gz
```

**(a)**

第一个 WARC response record 的 URL 是：

```text
http://0371rykj.com/ipfhsb/34.html
```

访问这个 URL 时会跳转到：

```text
http://www.0371rykj.com/ipfhsb/34.html
```

但是当前返回 `HTTP 502`，所以页面已经不能正常访问。直接看 raw HTML 可以看到大量 HTML entity 编码后的中文关键词、SEO metadata、CSS/JS 引用、隐藏标签和干扰性结构；页面主体看起来像工业设备产品页，但 title 和 metadata 混入了明显的 NSFW / spam 关键词，因此更像是被污染或 SEO spam 化的网页。

**(b)**

对应的第一个 WET record 也是同一个 URL。WET 文件确实抽出了可见文本，但抽取结果中仍然包含很多不是正文的内容，例如站点导航、产品分类、服务热线、上一篇/下一篇、页脚链接和产品参数表里的碎片化字段。

如果直接拿这种文本训练语言模型，模型会学习到很多网页模板、导航栏、SEO 关键词堆砌和乱码式编码噪声，而不是干净的自然语言。另一方面，这个页面里也有一部分关于恒温恒湿试验箱用途和参数的正文，如果能过滤掉 spam 和模板，只保留产品说明与参数，它仍然能提供一些中文工业设备领域的术语和格式信息。

**(c)**

这个样本在工业设备说明、B2B 产品页面理解这类场景下可能有一点价值，尤其是保留产品用途和参数表之后。但它不适合用于通用聊天助手、儿童安全场景或高质量中文语料构建，因为页面混有 NSFW spam、重复模板和营销联系方式。

**(d)**

继续查看后 25 个 WET records：

| 序号 | 域名 | 语言 | 页面类型 | 备注 |
| ---: | --- | --- | --- | --- |
| 2 | `10www.chinatikfans.com` | 中文 | Discuz 用户日志/论坛空间 | 登录、导航和论坛模板较多，正文少，质量偏低。 |
| 3 | `13.usnccm.org` | 英文 | 学术会议网站 | USNCCM 2015 会议页面，来源较可信，但菜单和站点结构占比较高。 |
| 4 | `176.utchat888.com` | 繁体中文 | 成人视频聊天室 | 成人内容和排行榜模板，应过滤。 |
| 5 | `176766.cn` | 中文 | 产品/资讯页 | 工业站点模板混入 NSFW SEO 关键词，质量很低。 |
| 6 | `178mh.com` | 中文 | 错误页 | 只有模板不存在提示，应过滤。 |
| 7 | `1796370.tgtg97.com` | 繁体中文 | 成人视频聊天室个人页 | 付费聊天、排行榜和重复模板很多，应过滤。 |
| 8 | `18sex.v340.info` | 繁体中文 | 成人视频站 | 成人内容、会员入口和排行榜模板，应过滤。 |
| 9 | `1kb.klimtoren.be` | 荷兰语 | 教师/班级博客 | 有少量真实博客内容，但 Blogger 模板较多。 |
| 10 | `1pekesat-exae.mysch.gr` | 希腊语 | helpdesk 搜索页 | 主要是论坛导航和搜索入口，训练价值较低。 |
| 11 | `1pekesat-exae.mysch.gr` | 希腊语 | helpdesk 登录页 | 登录页和论坛导航为主，质量低。 |
| 12 | `1s6605084.yhxzseo.com` | 中文 | SEO/博彩类资讯页 | 像自动生成的博彩 APP 推广文章，质量低。 |
| 13 | `20com20.fr` | 土耳其语 | Apache HTTP Server 文档 sitemap | 技术文档目录页，来源可信，结构清楚，是第一个相对高质量样本。 |
| 14 | `24ktcasino.net` | 英文 | casino tips 博客 | 有正文，但赌博主题和模板噪声较多。 |
| 15 | `2kgames.eu` | 英文 | 404 页面 | 只有 nginx 404，应过滤。 |
| 16 | `2l6185919.yizhangting.com` | 中文 | 博彩/健康资讯混合页 | 站点主题混乱，疑似 SEO 内容农场。 |
| 17 | `303323.com` | 中文 | 医疗器械公司新闻页 | 有真实产品新闻和联系方式，但模板较多。 |
| 18 | `30bad.com` | 中文 | 影视资源站条目页 | 影视条目和播放入口较多，正文价值一般。 |
| 19 | `312001.net` | 中文 | 社区卫生服务中心网站 | 医疗机构页面，可信但主要是导航和栏目列表。 |
| 20 | `354577.mwe075.com` | 繁体中文 | 成人视频聊天页 | 成人聊天室模板和排行榜，应过滤。 |
| 21 | `356.schoollibrary.edu.pe.ca` | 英文 | 学校图书馆检索结果页 | 查询字符串很长且无结果，对 LM 训练价值很低。 |
| 22 | `366392.haaxz.com` | 繁体中文 | 成人视频聊天页 | 成人内容和模板重复，应过滤。 |
| 23 | `366392.haaxz.com` | 繁体中文 | 成人视频聊天页 | 与上一条高度相似，重复性强，应过滤或去重。 |
| 24 | `387tel.com` | 繁体中文 | 视频聊天交友页 | 导航和排行字段为主，正文很少。 |
| 25 | `3diasdemarzo.blogspot.com` | 西班牙语 | 政治/新闻博客 | 有较长正文和引用，虽然有模板但整体质量较高。 |
| 26 | `3godetilbud.dk` | 丹麦语 | 本地服务 landing page | 商业服务落地页，有正文但营销模板明显。 |

第一个我认为可以算高质量的网页是第 `13` 个 WET record：`20com20.fr` 的 Apache HTTP Server 文档 sitemap。它虽然是文档目录而不是普通文章，但来源可信、文本结构清楚、噪声明显少于前面的成人站、错误页、登录页和 SEO 页面。如果标准放宽，第 `3` 个 `13.usnccm.org` 会议网站也可以算可用网页，但它的导航模板仍然比较多。

一句话总结：

> 前 26 条 Common Crawl WET records 中，低质量网页、成人内容、错误页、登录页、SEO spam 和模板重复非常常见；直到第 13 条左右才出现一个较可信的技术文档页面，这说明原始 Common Crawl 必须经过内容过滤和去重后才适合作为语言模型训练数据。

## 2.2 HTML to text conversion

**问题：extract_text**

使用 `resiliparse.extract.html2text.extract_plain_text` 从 WARC 的 HTML 中抽取纯文本。

在非 UTF-8 编码的页面上，先尝试常用编码（`utf-8`、`latin-1`、`cp1252`、`iso-8859-1`）解码，若均失败则用 `utf-8` + `errors="replace"` 兜底。

对应实现文件：

- `assignment4-data/cs336_data/extraction.py`
- `assignment4-data/tests/adapters.py`（`run_extract_text_from_html_bytes`）

**(a) 对比 WET 抽取 vs Resiliparse 抽取**

取第一个 WARC response，分别用 WET 自带文本和 Resiliparse 抽取文本做对比：

| | WET | Resiliparse (ours) |
| --- | ---: | ---: |
| 总字符数 | `3,496` | `10,165` |
| 总行数 | `127` | `993` |
| 空行数 | `1` | `744` |

Resiliparse 的输出比 WET 长了约 3 倍，行数是 8 倍，但其中约 75% 的行是空行。WET 抽取对空白和导航结构做了更激进的压缩，而 Resiliparse 保留了更多的 HTML 结构痕迹（缩进空行、列表项符号、隐藏标签中的文本片段等）。两者在同一页面上都保留了 NSFW/spam 关键词、产品名称、导航栏和服务热线等内容，在内容覆盖上大同小异，差异主要在格式和噪声密度上。

**(b) 使用我们自己的抽取器时要注意的问题**

第一个页面是 SEO spam 页面，Resiliparse 抽取时把 `display:none` 隐藏标签（如 `<xmp>`、`<menu>`、`<blockquote>` 等）里的文本也抽出来了，这些文本在浏览器中不可见，但对语言模型训练是纯噪声。实际抽取中可能需要先去掉 `display:none` 或 `hidden` 的元素再交给 Resiliparse。

另外，HTML entity 编码（如 `&#x4EBA;` 表示"人"）需要先被正确解码为 Unicode 字符。Resiliparse 内部使用 HTML 解析器处理了这个问题，所以我们的抽取结果中这些字符显示为正常中文，但如果自行实现简单正则抽取，就可能留下大量实体引用，污染文本。

**测试**

```bash
cd assignment4-data
python -m pytest tests/test_extract.py -v
```

结果：`1 passed`。

## 2.3 Language identification

**问题：language_identification**

使用 [fastText LID 模型](https://fasttext.cc/docs/en/language-identification.html) (`lid.176.ftz`) 做语言识别。模型文件放在：

```text
assignment4-data/cs336_data/assets/lid.176.ftz
```

直接使用 fastText 0.9.3 提供的 `.bin` 文件（120MB）在本环境里有兼容问题（无论输入什么文本都返回相同概率），改用 `.ftz` 量化版（917KB）后模型恢复正常工作。

对应实现文件：

- `assignment4-data/cs336_data/language.py`
- `assignment4-data/tests/adapters.py`（`run_identify_language`）

**测试**

```bash
python -m pytest tests/test_langid.py -v
# 2 passed
```

**(a) 语言识别错误对下游模型的影响**

语言识别错误主要有两种：false positive（非目标语言被保留）和 false negative（目标语言被丢弃）。前者会往训练语料中混入非英文/非目标语言的文本，导致模型在多语言混杂的 token 分布上训练，浪费容量并降低纯英文或纯中文场景的生成质量；后者会丢弃本可用的高质量文档，减少有效训练数据量。对于目标为英文的语言模型，最危险的是 false positive——把中文垃圾 SEO 页面、荷兰语模板、希腊语导航文本保留下来，因为它们虽然全是有效 Unicode 字符但训练中是不可消化的噪声。

**(b) 检查 20 个样本并对比人工判断与分类器预测**

在前 26 条 WET records 上对比我的人工判断和 fastText 预测：

| # | 人工判断 | fastText 预测 | 置信度 | 一致? |
| ---: | --- | --- | ---: | :---: |
| 1 | 中文 | `zh` | 0.83 | ✓ |
| 2 | 中文 | `zh` | 0.98 | ✓ |
| 3 | 英文 | `en` | 0.87 | ✓ |
| 4 | 繁体中文 | `zh` | 1.00 | ✓ |
| 5 | 中文 | `zh` | 0.95 | ✓ |
| 6 | 中文 | `zh` | 0.81 | ✓ |
| 7 | 繁体中文 | `zh` | 1.00 | ✓ |
| 8 | 繁体中文 | `zh` | 1.00 | ✓ |
| 9 | 荷兰语 | `nl` | 0.87 | ✓ |
| 10 | 希腊语 | `el` | 1.00 | ✓ |
| 11 | 希腊语 | `el` | 1.00 | ✓ |
| 12 | 中文 | `zh` | 1.00 | ✓ |
| 13 | 土耳其语 | `tr` | 0.96 | ✓ |
| 14 | 英文 | `en` | 0.95 | ✓ |
| 15 | 英文 | `fr` | 0.28 | ✗ |
| 16 | 中文 | `zh` | 1.00 | ✓ |
| 17 | 中文 | `zh` | 1.00 | ✓ |
| 18 | 中文 | `zh` | 1.00 | ✓ |
| 19 | 中文 | `zh` | 0.99 | ✓ |
| 20 | 繁体中文 | `zh` | 0.98 | ✓ |
| 21 | 英文 | `en` | 0.70 | ✓ |
| 22 | 繁体中文 | `zh` | 0.99 | ✓ |
| 23 | 繁体中文 | `zh` | 0.99 | ✓ |
| 24 | 繁体中文 | `zh` | 1.00 | ✓ |
| 25 | 西班牙语 | `es` | 0.95 | ✓ |
| 26 | 丹麦语 | `da` | 0.92 | ✓ |

26 条中只有 1 处不一致：第 15 条是一个 404 页面，文本内容仅 "404 Not Found / nginx"，fastText 预测为 `fr` 但置信度极低（0.28）。这说明极短文本或几乎空文本是语言识别的主要失败模式，而这类页面本来也应该被后续的质量过滤器或空文本过滤器去掉。

**(c) 英文文档比例估计与合适置信度阈值**

在这 26 条样本中，英文文档仅 3 条（第 3、14、21 条，其中第 21 条置信度 0.70 偏低），约 11.5%。如果以训练英文模型为目标，英文比例如此之低说明 Common Crawl 的全局语言分布高度偏向非英语内容，需要大量丢弃非英文页面。

如果把置信度阈值设为 `0.5`，26 条中唯一低于阈值的是第 15 条 404 页面（0.28），这意味着如果只靠置信度过滤，几乎所有非英文页面都会被保留，因为它们置信度都很高（中文页面普遍 0.95+，希腊语 1.00，土耳其语 0.96）。所以语言过滤的真正手段不是看置信度，而是直接按 `pred == "en"` 做二元分类，高置信度的中文/希腊语/土耳其语页面虽然预测正确，但同样应该丢弃。

## 2.4 Personal identifiable information

**问题：mask_pii**

Common Crawl 里会包含邮箱、电话、IP 地址等个人或机器标识信息。如果直接拿这些网页训练语言模型，模型可能记住并在生成时复现这些信息。因此本题不是删除整篇文档，而是把 PII 片段替换成固定占位符。

**1. Mask emails**

将 email address 替换为：

```text
|||EMAIL_ADDRESS|||
```

邮箱使用较标准的 email regex，匹配 `local@domain.tld` 形式，并返回替换后的文本和替换数量。

**2. Mask phone numbers**

将 phone number 替换为：

```text
|||PHONE_NUMBER|||
```

电话主要覆盖美国常见 10 位格式，例如 `2831823829`、`(283)-182-3829`、`(283) 182 3829`、`283-182-3829`。

**3. Mask IP addresses**

将 IPv4 address 替换为：

```text
|||IP_ADDRESS|||
```

IPv4 匹配 4 个 0-255 之间的 octets，并避免把非法地址替换掉。

三类 masking 对应关系：

| 类型 | 占位符 |
| --- | --- |
| email | `|||EMAIL_ADDRESS|||` |
| phone number | `|||PHONE_NUMBER|||` |
| IPv4 address | `|||IP_ADDRESS|||` |

对应实现文件：

- `assignment4-data/cs336_data/pii.py`
- `assignment4-data/tests/adapters.py`（`run_mask_emails`、`run_mask_phone_numbers`、`run_mask_ips`）

**测试**

```bash
cd assignment4-data
python -m pytest tests/test_pii.py -v
```

结果：`5 passed`。

**4. Naive PII masking 的下游风险**

Naive PII masking 的问题是会同时带来 false positives 和 false negatives。False positives 会把正常文本误替换掉，例如产品型号、备案号、文件 hash、日期或 QQ 号可能被 phone regex 误判；这会让训练文本出现过多占位符，模型可能学到不自然的 `|||PHONE_NUMBER|||` 模式。False negatives 则更危险，例如非美国电话格式、带国家区号的号码、中文座机号码、变体邮箱写法都可能漏掉，导致真实 PII 仍进入训练集。缓解方式是分地区设计 regex、结合上下文判断、先做规则过滤再做抽样人工审查，并在最终训练前统计每类占位符的出现频率。

**5. 在 WET 样本上的观察**

在前 1000 条 WET records 中，我找到前 20 个发生替换的文档，累计替换：

| 类型 | 替换次数 |
| --- | ---: |
| email | `15` |
| phone | `35` |
| IP | `0` |

观察到的合理替换包括：公司页面里的邮箱、英文/中文站点里的 `Email:` 字段、`Tel:`/`Fax:` 后面的电话号码，以及带国家区号的联系电话。也观察到一些明显 false positives：Rapidgator 文件 hash 中的一段数字被当成电话、ICP备案号或公安备案号被当成电话、`作者: cgf2020年...` 中的年份附近数字被误替换、casino 页面中表格数字被误认为电话。False negatives 也存在，例如 `0XXX-XXXXXXX` 这种中文座机格式没有被当前美国电话 regex 替换，`0XX XXX XXXXXX/XX` 这种带分机或 slash 的格式只会被部分替换。

一句话总结：

> PII masking 能降低模型记忆邮箱、电话和 IP 的风险，但规则过宽会破坏正常文本，规则过窄会漏掉真实 PII；实际数据处理中需要结合上下文、地区格式和抽样审查来调节 precision/recall。

## 2.5 Harmful content

**问题：harmful_content**

PII masking 是替换文档中的局部敏感片段，而 harmful content 这一节关注的是整篇文档是否应该被过滤。Common Crawl 中可能包含 NSFW 内容、辱骂、仇恨或 toxic speech；如果这些网页直接进入训练集，模型可能学到不适合用户-facing 产品的表达方式。本题使用 Dolma 提供的两个 Jigsaw fastText 分类器：一个判断 NSFW，一个判断 toxic speech。

**1. NSFW classifier**

使用官方要求的 Dolma fastText NSFW 模型：

```text
assignment4-data/cs336_data/assets/dolma_fasttext_nsfw_jigsaw_model.bin
```

函数 `run_classify_nsfw` 返回模型标签和置信度，标签为：

```text
nsfw / non-nsfw
```

**2. Toxic speech classifier**

使用官方要求的 Dolma fastText hatespeech/toxic 模型：

```text
assignment4-data/cs336_data/assets/dolma_fasttext_hatespeech_jigsaw_model.bin
```

函数 `run_classify_toxic_speech` 返回模型标签和置信度，标签为：

```text
toxic / non-toxic
```

对应实现文件：

- `assignment4-data/cs336_data/harmful.py`
- `assignment4-data/tests/adapters.py`（`run_classify_nsfw`、`run_classify_toxic_speech`）

**测试**

```bash
cd assignment4-data
python -m pytest tests/test_toxicity.py -v
```

结果：`2 passed`。

**3. Harmful content filters 的下游风险**

这类过滤器会改变训练数据的主题和表达分布。过滤太松时，模型仍会学到 NSFW、辱骂或攻击性语言；过滤太严时，模型可能丢掉合法讨论，例如医学、法律、新闻、历史、性教育、反骚扰讨论、毒性语言检测任务中的引用文本等。另一个问题是分类器本身有 domain shift：Jigsaw Wikipedia comments 上训练的模型不一定能准确识别网页广告、成人站模板、非英语文本或隐晦表达。缓解方法是把 fastText 分类器作为一个信号，而不是唯一规则；实际 pipeline 中可以结合关键词、URL/domain、语言过滤、质量过滤和人工抽样校准阈值。

**4. 在 WARC 抽取文本上的观察**

我在前 120 个 WARC response 中用 `extract_text_from_html_bytes` 抽取文本，然后随机抽 20 条跑 NSFW/toxic 分类。20 条中分类器预测的 harmful 比例为：

| 标签 | 数量 |
| --- | ---: |
| `nsfw` | `0 / 20` |
| `toxic` | `0 / 20` |

人工检查时发现，这 20 条中至少有几条来自明显成人或擦边站点，例如 `0371rykj.com`、`176.utchat888.com`、`387tel.com`，但 NSFW 分类器仍给出 `non-nsfw`，置信度还很高。这说明该 fastText 模型对 Jigsaw 风格英文 obscene comments 有 sanity-check 能力，但对中文/繁体中文成人站模板、SEO spam 和非英语网页的召回很差。随机样本中没有发现明显 toxic speech，被 toxic 模型全部判为 `non-toxic` 基本符合人工判断。

基于这个样本，我不会单独依赖 NSFW/toxic fastText 模型来过滤 Common Crawl。对于英文 comments 风格文本，可以使用较高阈值（例如 harmful label 且 confidence > `0.8`）过滤；但对网页数据，尤其是中文成人站和 SEO spam，还需要结合语言、关键词、URL/domain、Gopher quality rules 和质量分类器，否则 false negatives 会很多。

一句话总结：

> Dolma Jigsaw fastText 模型能通过官方 sanity tests，但在真实 Common Crawl 网页上存在明显 domain shift；它适合作为 harmful-content 信号之一，不适合作为唯一过滤器。

## 2.6 Quality Rules

**问题：gopher_quality_filters**

前面几节主要处理语言、PII 和 harmful content，但即使去掉这些问题，Common Crawl 中仍然有大量低质量网页，例如 404 页面、登录页、模板页、导航页、内容农场和抽取失败的页面。Gopher quality filters 的作用是用简单、可解释的规则先过滤掉一批明显不适合训练语言模型的文本。

**(a) 实现 Gopher quality filters 子集**

我实现了 handout 要求的四条规则：

| 规则 | 保留条件 |
| --- | --- |
| 文档长度 | 词数在 `50` 到 `100000` 之间 |
| 平均词长 | mean word length 在 `3` 到 `10` 之间 |
| 省略号行比例 | 以 `...` 结尾的行不超过 `30%` |
| 含字母词比例 | 至少 `80%` 的词含有 alphabetic character |

对应实现文件：

- `assignment4-data/cs336_data/quality.py`
- `assignment4-data/tests/adapters.py`（`run_gopher_quality_filter`）

这里没有使用 NLTK，而是用 whitespace tokenization。这个选择更简单，也避免了额外 tokenizer 资源下载；对本题要求的规则来说已经足够。

**测试**

```bash
cd assignment4-data
python -m pytest tests/test_quality.py -k gopher -v
```

结果：`7 passed`。

**(b) 在 WARC 抽取文本上的观察**

我在前 160 个 WARC response 中用 `extract_text_from_html_bytes` 抽取文本，然后随机抽 20 条跑 Gopher quality filter。20 条中有 6 条通过，14 条被过滤。

| # | 域名 | Gopher | 人工判断 | 备注 |
| ---: | --- | :---: | --- | --- |
| 2 | `10www.chinatikfans.com` | reject | reject | 编码乱码，平均词长异常，alpha 比例低。 |
| 145 | `artoffiction.blogspot.com` | reject | keep-ish | 有真实文章内容，但被 alpha 比例误伤。 |
| 50 | `aawptraining.com` | keep | reject | WordPress 培训站模板和导航很多，规则放过。 |
| 37 | `7bergencycling.nl` | keep | keep-ish | 俱乐部页面，有可读文本但模板较多。 |
| 147 | `ask.mallaky.com` | reject | reject | hosting unavailable 页面，太短。 |
| 97 | `altitude-creative.au` | reject | reject | landing page 导航和服务列表为主，alpha 比例略低。 |
| 47 | `a16.wahas.com` | reject | reject | 成人/影视论坛页面，含字母比例低。 |
| 122 | `arandatres.onlinewebshop.net` | reject | reject | 商店模板混合文本，噪声较多。 |
| 7 | `1796370.tgtg97.com` | reject | reject | 编码乱码，成人聊天模板，平均词长异常。 |
| 135 | `arellanos.blogspot.com` | reject | keep-ish | 有西语博客内容，但导航和模板较多，alpha 比例低。 |
| 146 | `ashishtradecentre.com` | reject | reject | 商业站点栏目/商品列表为主。 |
| 91 | `all-noise.co.uk` | keep | keep | 音乐站点，有正常正文和列表。 |
| 17 | `303323.com` | reject | reject | HTML 表单残留和抽取噪声很多，平均词长偏高。 |
| 114 | `apidc.org` | keep | keep | board game review 站点，文本较自然。 |
| 23 | `366392.haaxz.com` | reject | reject | 编码乱码，成人聊天模板。 |
| 136 | `ariel5437.pixnet.net` | reject | reject | 图片相册页，正文很少。 |
| 115 | `apidc.org` | keep | keep | 同一站点的 review 页面，质量较好。 |
| 155 | `asu.tgizd.ru` | keep | reject | 明显编码损坏，但规则放过。 |
| 79 | `aichi-progolf.info` | reject | reject | 导航和联系方式为主，日英混合模板。 |
| 46 | `9se843.xyz` | reject | reject | 成人站/视频站模板，alpha 比例低。 |

主要差异有两类。第一类是 false positives：`artoffiction.blogspot.com`、`arellanos.blogspot.com` 这类博客有真实正文，但由于抽取结果混入大量导航、模板或非英语字符，含字母词比例低于阈值而被过滤。第二类是 false negatives：`aawptraining.com` 和 `asu.tgizd.ru` 通过了规则，但人工看一个像 WordPress 模板页，另一个明显有编码损坏。也就是说，Gopher rules 对长度、符号比例和极端格式问题很有效，但无法可靠判断语义质量、主题质量和编码是否损坏。

一句话总结：

> Gopher quality filters 是便宜、可解释的第一道质量闸门，能去掉短页、乱码页、成人模板页和符号占比异常的页面，但它不是完整质量判断器；真实 pipeline 里还需要语言过滤、harmful-content 过滤、去重和质量分类器共同使用。

## 2.7 Quality Classifier

**问题：quality_classifier**

Gopher quality filters 只能捕捉长度、词长、符号比例这类表面特征，不能可靠判断一篇网页是否像高质量参考资料。quality classifier 的目标是学习高质量 reference pages（题目建议用 Wikipedia 外链页面）和普通 Common Crawl 页面之间的差异，然后给页面一个质量标签或质量分数。

**(a) 训练质量分类器**

按作业设计，质量分类器应该用 Wikipedia external links 抓取页面作为 high-quality 正例，用 Common Crawl 页面作为负例，然后训练一个 fastText 二分类器。Wikipedia reference URL 列表来自 handout 给出的地址：

```text
https://nlp.stanford.edu/data/nfliu/cs336-spring-2024/assignment4/enwiki-20240420-extracted_urls.txt.gz
```

我已将该文件下载到：

```text
assignment4-data/data/enwiki-20240420-extracted_urls.txt.gz
```

正式训练流程如下。由于直接用单个 `wget` 串行抓取 `10000` 个 Wikipedia reference URLs 速度太慢，我最终使用 16 个 shard 并行抓取，然后把 16 个 WARC shards 一起交给训练脚本：

```bash
cd assignment4-data

bash scripts/run_quality_10000_parallel.sh
```

我先训练过一个 smoke 模型来保证代码路径真实走 fastText classifier，而不是只用手写 heuristic：

```bash
python scripts/train_quality_classifier.py \
  --smoke-fixtures \
  --epoch 50 \
  --lr 0.8 \
  --word-ngrams 2
```

正式 `10000` URL 版本已完成。运行日志和自动 summary 分别在：

```text
assignment4-data/data/quality_full_10000_parallel_run.log
assignment4-data/data/quality_full_10000.summary.txt
```

完成时间：`2026-05-24T10:21:04+08:00`。16 个 WARC shards 全部抓取完成，单 shard 大小从 `22M` 到 `99M` 不等。训练脚本对 Wikipedia reference pages 做 HTML text extraction，并且用 `--apply-gopher` 先过滤低质量正例；因此虽然采样了 `10000` 个 Wikipedia reference URLs，最终可用正例少于 10000。

最终训练数据规模：

| label | examples |
| --- | ---: |
| `wiki` | `3826` |
| `cc` | `10000` |

正式训练输出：

```text
assignment4-data/cs336_data/assets/quality_classifier.bin
assignment4-data/data/quality_classifier.full_10000.train.txt
```

模型文件大小约 `1.4G`，训练文本大小约 `104M`。

`classify_quality` 会优先加载该 fastText 模型；只有模型文件不存在时才 fallback 到启发式分类器。

对应实现文件：

- `assignment4-data/cs336_data/quality.py`（`classify_quality`）
- `assignment4-data/scripts/sample_wiki_urls.py`
- `assignment4-data/scripts/train_quality_classifier.py`
- `assignment4-data/scripts/run_quality_10000_parallel.sh`
- `assignment4-data/tests/adapters.py`（`run_classify_quality`）

**(b) classify_quality 接口**

函数返回：

```text
(label, confidence)
```

其中 label 为：

```text
wiki / cc
```

在官方 sanity fixtures 上，当前 fastText 模型输出：

| fixture | top-1 预测 | confidence |
| --- | --- | ---: |
| `low_quality_cc.txt` | `cc` | `1.000` |
| `high_quality_wiki_reference.txt` | `wiki` | `0.723` |

**测试**

```bash
cd assignment4-data
python -m pytest tests/test_quality.py -v
```

结果：`8 passed`。

完整测试也通过：

```bash
python -m pytest -v
```

结果：`21 passed`。

2026-06-16 在 `coding` 环境复测时，fastText 0.9.x 与 NumPy 2 的 `np.array(..., copy=False)` 行为不兼容会导致 language/quality/toxicity 测试失败。已加入 `cs336_data.fasttext_compat.predict_top1`，直接使用 fastText 底层 binding 返回 top-1 预测，保留原有输出格式。复测命令：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
python -m pytest -q -p no:cacheprovider
```

结果：`21 passed`。

一句话总结：

> `classify_quality` 现在加载的是正式 `10000` URL 流程训练出的 fastText 二分类器；Wikipedia reference 正例经过抓取、HTML 抽取和 Gopher 过滤后得到 `3826` 条，Common Crawl 负例取满 `10000` 条，官方质量测试和完整测试均通过。

## 2.8 Exact Deduplication

**问题：exact_deduplication**

Common Crawl 里有大量重复内容。重复可能来自同一个网页被多次抓取、镜像站、模板页、页脚、导航栏、版权声明、广告片段等。Exact line deduplication 处理的是最简单的一类重复：完全相同的文本行。如果一行在整个 corpus 中出现多次，就把它从所有输出文档中删除。

**实现方式**

我使用两遍扫描：

1. 第一遍读取所有输入文件，统计每一行在整个输入集合中的出现次数。
2. 第二遍重新读取每个输入文件，只保留全局出现次数为 `1` 的行。
3. 输出目录中保留和输入文件相同的 basename。

对应实现文件：

- `assignment4-data/cs336_data/deduplication.py`（`exact_line_deduplication`）
- `assignment4-data/tests/adapters.py`（`run_exact_line_deduplication`）

这个实现故意使用 line-level exact match，不做 normalization。也就是说，只有字节/字符串层面完全相同的行才会被删除；如果两行只是在大小写、空白、标点上略有差异，则不会被 exact dedup 去掉，后面需要 MinHash/LSH 处理。

**测试**

```bash
cd assignment4-data
python -m pytest tests/test_deduplication.py::test_exact_line_deduplication -v
```

结果：`1 passed`。

一句话总结：

> Exact line deduplication 是便宜、确定性的第一层去重：它能删除全 corpus 中重复出现的完全相同行，但不能处理 paraphrase、模板轻微变化或整篇文档级近似重复。

## 2.9 MinHash Deduplication

**问题：minhash_deduplication**

Exact line dedup 只能删除完全相同的行，不能处理整篇文档级别的近似重复。Common Crawl 中常见的重复更复杂：同一个 license 文本可能只改了版权方，镜像页面可能只改了导航栏，模板页面可能只改了少量字段。MinHash + LSH 的目标是把这些 Jaccard 相似度很高的文档聚到同一个 duplicate cluster 中，并且每个 cluster 只保留一个代表文档。

**实现方式**

我实现的流程如下：

1. 对每篇文档做 Unicode normalization、去重音、lowercase、去标点和 whitespace normalization。
2. 用 word-level `n`-grams 构造 shingle set；测试中使用 `ngrams=5`。
3. 对每个 shingle set 计算 `num_hashes` 维 MinHash signature。
4. 把 signature 切成 `num_bands` 个 band，LSH bucket 命中的文档对作为候选近重复。
5. 对候选对计算真实 Jaccard similarity，达到 `jaccard_threshold` 才 union 到同一个 cluster。
6. 用 union-find 合并 duplicate cluster，并保留每个 cluster 中输入顺序最靠前的文件。

对应实现文件：

- `assignment4-data/cs336_data/deduplication.py`（`minhash_deduplication`）
- `assignment4-data/tests/adapters.py`（`run_minhash_deduplication`）

这个实现把 LSH 只作为 candidate generation；最终是否去重仍然用真实 Jaccard similarity 判定，因此不会仅因为某个 band 碰撞就删除文档。测试里的 fuzzy duplicate 是两份 MIT license 文本，它们在版权行和空白上有差异，但主体 license 文本高度相同，应该只保留其中一份。

**测试**

```bash
cd assignment4-data
python -m pytest tests/test_deduplication.py -v
```

结果：`3 passed`。

完整测试：

```bash
python -m pytest -v
```

结果：`21 passed`。

一句话总结：

> MinHash + LSH 是 exact dedup 之后的近重复过滤层：它通过 word n-gram Jaccard similarity 捕捉整篇文档级别的相似文本，并用 LSH 避免对所有文档两两比较。

## 2.10 Filter Data

**问题：filter_data**

这一节的目标不是再实现单个 filter primitive，而是把前面几节的过滤器组合成一个能处理 Common Crawl WET 文件的 language-modeling data pipeline。Leaderboard 的评价方式是：固定模型结构和训练流程，只改变训练数据，然后看 GPT-2 small-shaped 模型在 Paloma C4 100 domains validation set 上的 perplexity。因此这里的重点是数据选择，而不是改模型。

**实现脚本**

我实现了：

```text
assignment4-data/scripts/filter_data.py
```

脚本输入一个或多个 `.warc.wet.gz` 文件，输出：

```text
<output-dir>/text/*.filtered.txt.gz
<output-dir>/filter_stats.json
<output-dir>/samples/kept/*.jsonl
<output-dir>/samples/rejected/*.jsonl
```

其中 `text/*.filtered.txt.gz` 是最终 LM training text，每篇文档后写入 GPT-2 常用的 document boundary：

```text
<|endoftext|>
```

脚本支持 `--workers` 用 `ProcessPoolExecutor` 并行处理多个 WET 文件，也支持 `--max-docs-per-file` 做快速实验。由于 `quality_classifier.bin` 约 `1.4G`，多进程同时加载质量分类器会占用很多内存，所以正式大规模运行时要么降低 `--workers`，要么先不用该质量分类器，转而使用更轻量的规则和后续人工检查调参。

**当前 pipeline**

当前 baseline pipeline 按顺序执行：

1. 读取 WET conversion record，并做 whitespace normalization。
2. 删除空文档、过短文档和过长文档。
3. 用 URL/domain pattern 删除明显成人、博彩和 spam 域名。
4. 用 text pattern 删除明显低价值页面，例如 sex cams、casino、forum registration agreement。
5. 用 fastText `lid.176.ftz` 只保留英文，阈值为 `0.65`。
6. 用 Gopher quality rules 删除长度、词长、符号比例或 alphabetic word 比例异常的文本。
7. 用 Dolma/Jigsaw NSFW 和 toxic fastText 模型删除高置信 harmful 文档，阈值为 `0.8`。
8. 对保留文本做 email、phone、IP masking。
9. 输出过滤后的文档，并记录 kept/rejected reservoir samples 供人工检查。

我实验过把第 2.7 节训练出的 `wiki` vs `cc` quality classifier 加入 pipeline，但在本地前 `500` 条 WET 样本上它过严：`500` 条中 `0` 条最终保留，`97` 条被 quality classifier 拒绝。因此当前 baseline 关闭该分类器：

```bash
--no-quality-classifier
```

这个选择的原因是 Paloma C4 100 domains 更接近普通网页，而不是 Wikipedia reference pages；wiki-vs-CC 分类器适合做高精度质量信号，但直接作为 hard filter 会把很多 C4 风格普通网页也删掉。

**本地完整 WET 运行**

由于当前机器没有 `/data/CC/CC*.warc.wet.gz` 这批 5000 个 cluster 文件，我在本地已下载的一个 WET 文件上跑完整 baseline：

```text
assignment4-data/data/CC-MAIN-20250417135010-20250417165010-00065.warc.wet.gz
```

运行命令：

```bash
cd assignment4-data

python scripts/filter_data.py \
  --input data/CC-MAIN-20250417135010-20250417165010-00065.warc.wet.gz \
  --output-dir data/filtered_one_wet_v2 \
  --workers 1 \
  --sample-limit 100 \
  --no-quality-classifier
```

该 WET 文件压缩大小约 `80.9M`，包含 `27173` 个 conversion records。运行结果：

| step / reason | count | fraction of raw |
| --- | ---: | ---: |
| raw | `27173` | `100.00%` |
| language | `16231` | `59.73%` |
| kept | `8677` | `31.93%` |
| too_short | `801` | `2.95%` |
| gopher | `701` | `2.58%` |
| domain_blocklist | `386` | `1.42%` |
| low_value_pattern | `323` | `1.19%` |
| too_long | `22` | `0.08%` |
| nsfw | `17` | `0.06%` |
| toxic | `15` | `0.06%` |

PII masking 修改了：

| PII type | replacements |
| --- | ---: |
| email | `3451` |
| phone | `6077` |
| IP | `233` |

输出文件：

```text
assignment4-data/data/filtered_one_wet_v2/text/CC-MAIN-20250417135010-20250417165010-00065.warc.wet.gz.filtered.txt.gz
```

输出压缩大小约 `17.9M`。

**运行时间估计**

本地单 worker 处理一个约 `80.9M` 的 WET 文件用时 `33.99s`。Handout 说 5000 个 WET 文件约 `375GB` 压缩文本，平均每个约 `75MB`，和本地样本大小接近。因此粗略估计：

| scale | single worker estimate | 16 workers idealized estimate |
| --- | ---: | ---: |
| 5000 WETs | `~47.2h` | `~3.0h` |
| 100000 WETs | `~39.3d` | `~2.5d` |

这个估计假设 WET 文件大小、I/O 带宽和 CPU 负载接近线性；真实 cluster 上会受共享文件系统、模型加载、任务调度和单文件大小差异影响。

一句话总结：

> 当前 `filter_data.py` baseline 用英文识别、Gopher rules、安全过滤、URL/text blocklist 和 PII masking 生成 C4 风格网页训练文本；wiki-vs-CC quality classifier 作为可选高精度 filter 保留，但当前不作为 hard filter 使用，因为它对普通 CC/C4 风格网页过严。

## 2.11 Inspect Filtered Data

**问题：inspect_filtered_data**

我实现了 inspection 脚本：

```text
assignment4-data/scripts/inspect_filtered_data.py
```

它读取 `filter_data.py` 输出的 kept/rejected sample JSONL，并生成 Markdown 报告：

```bash
cd assignment4-data

python scripts/inspect_filtered_data.py \
  --filter-output-dir data/filtered_one_wet_v2 \
  --output data/filtered_one_wet_v2/inspection.md \
  --num-examples 5 \
  --max-chars 900
```

报告位置：

```text
assignment4-data/data/filtered_one_wet_v2/inspection.md
```

**(a) 五个保留样本**

| URL | 人工评价 |
| --- | --- |
| `ipfonline.com/.../fabricated-flanges-799-price` | 工业产品页面，包含大量导航和分类，但也有产品信息；对 C4 风格网页建模可用但模板噪声偏多。 |
| `lensntrends.com/products/versace-ve2272-sunglasses` | 电商商品页，导航和品牌列表很多，正文价值一般；保留后可能让模型学习电商模板。 |
| `peek.com/.../utv-and-jeep-island-adventure` | 旅游活动页，包含活动描述、评分、时长和 cancellation 信息；比较接近 C4 普通网页，适合训练。 |
| `api.dart.dev/.../ByteBuffer/asInt32x4List.html` | Dart API 文档，结构清楚、技术内容密集，是高质量 LM 训练文本。 |
| `timeforsuccess.io/reading-list` | 商业读书清单页面，PII masking 生效；正文是书目列表，价值中等。 |

保留样本说明当前 pipeline 能留下技术文档、博客、旅游页、电商页和列表页，整体比 raw WET 干净；但仍有模板噪声，例如导航栏、商品分类和重复 CTA。后续如果目标是更低 Paloma loss，可以增加 boilerplate removal 或降低模板页权重。

**(b) 五个丢弃/修改样本**

| URL | reason | 人工评价 |
| --- | --- | --- |
| `realpraktic.sk/.../200-l` | `language=sk` | 斯洛伐克语电商页，英文模型目标下删除合理。 |
| `papir.cehr.ft.ucp.pt/...` | `language=pt` | 葡萄牙语档案浏览页，模板较多；删除合理。 |
| `charriol.com/ar-qa/...` | `language=ar` | 页面有英文片段但 URL locale 和重复 copyright 结构明显异常；删除合理。 |
| `hvs-rezultati.furkisport.com/...ln=hr` | `language=sr` | 体育比赛结果页，文本短且非英语；删除合理。 |
| `bookwa.org/legalizaciya-dokumentov-v-mid.html` | `language=ru` | 俄语商业/法律文章页；删除合理。 |

PII masking 也会修改保留文本。例如 `timeforsuccess.io/reading-list` 中电话被替换成 `|||PHONE_NUMBER|||`。这类修改合理，因为它降低了模型记忆联系方式的风险。

**(c) 迭代结论**

第一次不使用额外 blocklist 的 baseline 在前 `500` 条样本中误保留了成人 webcam 页面和论坛注册协议页面。根据 inspection，我加入了：

1. URL/domain blocklist：匹配 `sex`、`porn`、`casino`、`escort`、`adult`、`webcam`、`cam2cam`、`xxx` 等模式。
2. Text low-value pattern：匹配 `free sex cams`、`nude webcam`、`private cams`、`casino`、`forum registration agreement`、`log in to check your private messages` 等明显低价值模板。
3. 保留 wiki-vs-CC quality classifier 作为可选项，但默认实验关闭，因为它在本地样本上过严。

一句话总结：

> Inspection 显示当前 pipeline 的主要问题不是非英语或 harmful 内容，而是模板噪声和商业列表页；后续提升 leaderboard 表现的方向应是更精细的 boilerplate removal、domain-aware sampling 和用 Paloma C4 validation 风格调节 quality threshold。

## 2.12 Tokenize Data

**问题：tokenize_data**

在训练语言模型之前，需要把过滤后的文本数据转换成模型能直接使用的 token ID 序列。`tokenize_data` 的任务是用 GPT-2 tokenizer 将 `filter_data.py` 输出的文档编码为整数 ID，并序列化成训练脚本能读取的二进制格式。

**实现脚本**

```text
assignment4-data/scripts/tokenize_data.py
```

脚本流程：

1. 读取 `filter_data.py` 输出的 `.txt.gz` 文件。
2. 按 `<|endoftext|>` 分隔符拆分成独立文档。
3. 用 GPT-2 tokenizer（`AutoTokenizer.from_pretrained("gpt2")`）将每篇文档编码为 token ID 列表。
4. 每篇文档末尾附加 `eos_token_id`（`50256`），作为训练时的 document boundary。
5. 将所有文档的 token ID 展平成一个一维 `np.uint16` 数组。
6. 用 `tofile()` 写入二进制文件，供 `train.py` 通过 `np.memmap(..., dtype=np.uint16)` 加载。

二进制格式必须严格遵循 handout 示例代码，否则 `cs336-basics/scripts/train.py` 无法加载：

```python
ids_array = np.array(all_ids, dtype=np.uint16)
ids_array.tofile(output_path)
```

训练脚本读取方式：

```python
train_data = np.memmap(cfg.paths.train_bin, dtype=np.uint16, mode="r")
```

**本地运行**

```bash
cd assignment4-data
python scripts/tokenize_data.py \
  --input data/filtered_one_wet_v2/text/CC-MAIN-20250417135010-20250417165010-00065.warc.wet.gz.filtered.txt.gz \
  --output data/filtered_one_wet_v2/tokenized.bin \
  --workers 1
```

运行结果：

| 指标 | 值 |
| --- | ---: |
| input documents | `8677` |
| output tokens | `13,962,921` |
| output file size | `27,925,842 bytes` |
| dtype | `np.uint16` |
| tokenizer | `gpt2` |

验证：用 `np.memmap(..., dtype=np.uint16)` 读取后，前 50 个 token 解码为 "Welcome to USNCCM13! | USNCCM 13..."，确认 GPT-2 编码正常。

**config 配置**

要把 tokenized data 用于训练，需要修改：

```text
assignment4-data/cs336-basics/configs/experiment/your_data.yaml
```

将其中的 `paths.train_bin` 指向 tokenized 二进制文件，例如：

```yaml
paths:
  train_bin: /mdata/wjx/CS336/assignment4-data/data/filtered_one_wet_v2/tokenized.bin
```

一句话总结：

  > `tokenize_data` 把过滤后的文本文档用 GPT-2 tokenizer 编码为 `np.uint16` 二进制文件，包含 EOS token 作为文档边界，输出格式与 provided training script 兼容。

## 2.13 Train Model

**问题：train_model**

这一步使用作业提供的训练脚本 `cs336-basics/scripts/train.py` 在 tokenized 数据上训练 GPT-2 small-shaped 模型，并按固定间隔在 Paloma C4 100 domains validation set 上评估 perplexity。目标是让模型在该验证集上得到尽可能低的 validation loss，并且不能修改模型结构或训练过程。

**配置**

我修改了 `cs336-basics/configs/experiment/your_data.yaml`，将 `paths.train_bin` 指向本地 tokenized 二进制文件：

```yaml
paths:
  train_bin: assignment4-data/data/filtered_one_wet_v2/tokenized.bin
  valid_bin: assignment4-data/data/filtered_one_wet_v2/valid.bin
  model_output: output/your_data

training:
  wandb_entity: null
  wandb_project: null
```

注意：`valid_bin` 在本机环境中指向从训练数据切分出的 synthetic validation set，因为真正的 Paloma C4 100 domains validation data 仅存在于 Together cluster（`/data/paloma/tokenized_paloma_c4_100_domains_validation.bin`）。在 cluster 上正式提交 leaderboard 时，应将其指向该文件。

**模型规格**

GPT-2 small-shaped（124M parameters）：

| 参数 | 值 |
| --- | ---: |
| vocab_size | `50257` |
| context_length | `512` |
| d_model | `768` |
| num_layers | `12` |
| num_heads | `12` |
| d_ff | `2048` |
| non-embedding params | `84.95M` |

训练配置：

| 参数 | 值 |
| --- | ---: |
| batch size per device | `32` |
| gradient accumulation | `1` |
| effective batch | `32 × 512 = 16384 tokens/step` |
| dtype | `bfloat16` |
| optimizer | AdamW, lr `1e-3`, cosine schedule |
| weight decay | `0.1` |
| max grad norm | `1.0` |
| `torch.compile` | disabled for smoke run |

注意：正式训练应使用 `batch_size=128`、`gradient_accumulation_steps=1`、2 GPU DDP（`torchrun --nproc_per_node=2`），共 `128 × 512 × 2 = 131072 tokens/step`。本机因单 GPU 内存限制（lm_head 输出 `[batch*seq, vocab]` 在 128 batch 时 `6.1 GiB`，加上模型和 optimizer state 后超过 32GB），使用 batch 32 验证流程。

**运行命令**

```bash
cd assignment4-data/cs336-basics
python scripts/train.py --config-name=experiment/your_data \
  +training.train_steps=2000 \
  +training.eval_interval=200 \
  +training.eval_iterations=50 \
  +training.compile=false \
  +training.save_checkpoints=false \
  +training.wandb_project=null \
  +training.wandb_entity=null \
  +training.train_batch_size=32 \
  +training.gradient_accumulation_steps=1
```

正式 cluster 命令：

```bash
cd assignment4-data
uv run torchrun --standalone --nproc_per_node=2 cs336-basics/scripts/train.py \
  --config-name=experiment/your_data
```

**训练结果**

单 GPU 单 WET 训练数据（13.96M tokens），2000 steps：

| step | train loss | val loss (eval) |
| ---: | ---: | ---: |
| 0 | `10.83` | — |
| 200 | — | `7.47` |
| 400 | — | `6.20` |
| 600 | — | `5.20` |
| 800 | — | `4.67` |
| 1000 | — | `4.21` |
| 1200 | — | `4.00` |
| 1400 | — | `3.71` |
| 1600 | — | `3.58` |
| 1800 | — | `3.56` |
| 2000 | `3.58` | `3.58` |

最佳 validation loss：约 `3.56`（step 1800）。Final validation loss：`3.58`。

**注意事项**

1. 本次训练的 validation set 是训练数据的一部分，不是独立的 Paloma C4 100 domains 数据。Paloma validation 文件在 Together cluster 上，Leaderboard 提交必须使用真实文件。
2. 训练数据仅来自 1 个 WET 文件（约 14M tokens），远小于正式任务要求的 5000 个 WET 文件。因此 validation loss 绝对值不代表 leaderboard 分数。
3. DDP 2 GPU 在本机因 NCCL 初始化问题未启用；cluster 上用 `torchrun --nproc_per_node=2` 即可正常启动。
4. `torch.compile` 在本次 run 中关闭以加快实验迭代；正式训练应开启以获得更好的吞吐量。
5. 训练产物 `output/your_data/model.pt` 和 `model_config.json` 已本地保存。

一句话总结：

> `train_model` 使用 provided GPT-2 small-shaped training script 在 tokenized 过滤数据上完成了 2000-step 训练，最佳 validation loss 约 `3.56`；完整 leaderboard 训练需要在 cluster 上用全部 5000 WET 文件和真实 Paloma validation set 跑 200K steps。
