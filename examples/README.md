# Examples

`sample-docs/` is a tiny corpus about a fictional coffee roastery: a refund policy, shipping terms, a safety manual, an employee handbook and a set of meeting notes. It is enough to try every search strategy in a minute.

```bash
contextsearch index examples/sample-docs
contextsearch search "how do I get my money back" --strategy hybrid
contextsearch eval examples/labels.json --strategies lexical,dense,hybrid,hybrid+rerank
```

`labels.json` lists queries with the documents that should come back. `contextsearch eval` scores each strategy on it; copy the format to build a labels file for your own corpus, which is the reliable way to decide between models and settings.
