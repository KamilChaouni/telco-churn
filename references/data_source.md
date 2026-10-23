# Data source record — IBM Telco Customer Churn

## Identity of the raw snapshot

| Item | Value |
|---|---|
| Publisher | IBM, repository `IBM/telco-customer-churn-on-icp4d` |
| Pinned commit | `1fd6fd70906479be1712e64cfc0ea89f58466629` |
| Download URL | https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/1fd6fd70906479be1712e64cfc0ea89f58466629/data/Telco-Customer-Churn.csv |
| Local path | `data/raw/telco.csv` (not in Git) |
| Size | 970 457 bytes, 7 044 lines (header + 7 043 customers), LF line endings, no BOM |
| SHA-256 | `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91` |

### Obtain and verify

```sh
curl -L -o data/raw/telco.csv "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/1fd6fd70906479be1712e64cfc0ea89f58466629/data/Telco-Customer-Churn.csv"
sha256sum data/raw/telco.csv          # Linux
shasum -a 256 data/raw/telco.csv      # macOS
Get-FileHash data/raw/telco.csv       # Windows PowerShell
```

The inspection report also recomputes the hash and sets `source.matches_expected_sha256`.
Never open and re-save this file in a spreadsheet application.

## Incident: the copy received in class (`datasets.csv`) is NOT byte-identical

| | Received `datasets.csv` | Pinned original |
|---|---|---|
| SHA-256 | `62ca518b1ebbcd4c832c18c1a21629b03abe9033fef66fe2e38d8d030658a20c` | `16320c9c…055e91` |
| Encoding | UTF-8 **with BOM** | UTF-8, no BOM |
| Line endings | CRLF | LF |
| Quoting | each whole line wrapped in `"…"` | none |
| Parsed by `pd.read_csv` | **1 column** (`customerID,gender,…`) | 21 columns |

After removing the BOM, the CRLF and the wrapping quotes, all 7 044 lines are identical
to the original: the business content is the same but the bytes (and so the hash) are not.
This looks like the file was opened and re-saved by a spreadsheet tool. We therefore use
the pinned original as `data/raw/telco.csv`; the inspection program refuses the received
copy with an explicit error instead of producing a misleading report.

## Field roles

| Role | Columns |
|---|---|
| Identifier (never a model input) | `customerID` |
| Target (never a model input) | `Churn` (`Yes` / `No`) |
| Numeric inputs | `tenure`, `MonthlyCharges`, `TotalCharges` (read as text, see below), `SeniorCitizen` (0/1 flag) |
| Categorical inputs | `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod` |

## Field notes (from `reports/data_overview.json`)

- 7 043 rows × 21 columns, 0 duplicated `customerID`, 0 `NaN` reported by `isna()`.
- `TotalCharges` is read as **text** because 11 values are a single space `" "`.
  `isna()` does not see them; the whitespace check does. All 11 have `tenure == 0`
  (new customers not yet billed). Handling them is a TP02 preprocessing decision.
- Target: `No` 5 174 (73.46 %), `Yes` 1 869 (26.54 %) — imbalanced.
- This is a single customer snapshot: no dates, no event history, no randomized
  retention treatment. It supports association/prediction, not causal claims.
