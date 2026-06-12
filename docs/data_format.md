# Data Format

The loader accepts CSV and whitespace-delimited TXT files. Common field and resistance column names are normalized to:

```text
B, Rxx, Rxy
```

Recognized examples include:

```text
B, Rxx, Rxy
Field_T, Rxx_Ohm, Rxy_Ohm
field, rhoxx, rhoxy
```

Loaded DataFrames store import metadata in:

```python
frame.attrs["transport_metadata"]
```
