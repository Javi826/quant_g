# DEMO MODE REMOVAL GUIDE
## Complete Guide to Remove Demo Trading Mode from BOT_trading

**Date:** March 2026  
**Version:** 1.0  
**Purpose:** High-level instructions to remove all demo mode functionality from the trading bot

---

## 🎯 OVERVIEW

Demo mode was implemented to validate LAB backtests against live market conditions without:
- Executing real broker orders
- Writing to PostgreSQL database
- Affecting production accounts

**If you decide to remove demo mode**, this guide provides the roadmap.

---

## 📋 FILES TO MODIFY

### **Core Module Files:**

1. **`core/demo_operative.py`**
   - **Action:** DELETE entire file
   - **Description:** Contains DemoOperative class with simulated order execution

2. **`core/orchestrator.py`**
   - **Action:** MODIFY (remove demo integration)
   - **Lines affected:** ~150-200, ~270-280, ~760-775
   - **Changes needed:**
     - Remove `demo_operative` attribute initialization
     - Remove demo mode detection logic in `__init__`
     - Remove conditional demo monitoring in `_periodic_tpsl_check()`
     - Remove demo order placement hooks

3. **`config/settings.py`**
   - **Action:** MODIFY (remove demo configuration)
   - **Lines affected:** ~180-185
   - **Changes needed:**
     - Remove `DEMO_MODE_ACCOUNTS` list/set
     - Remove any demo-specific settings

---

### **State Management Files:**

4. **`state/state_manager.py`**
   - **Action:** MODIFY (remove demo bypass logic)
   - **Lines affected:** ~50-60 in `save_state_local()`
   - **Changes needed:**
     - Remove early return check for demo accounts
     - Remove demo mode import from settings
     - Restore uniform state persistence for all accounts

---

### **Strategy Processing Files:**

5. **`strategies/strategy_processor.py`**
   - **Action:** MODIFY (remove demo order routing)
   - **Lines affected:** ~120-180 in `process()` method
   - **Changes needed:**
     - Remove demo mode detection
     - Remove calls to `demo_operative.place_simulated_order()`
     - Restore direct calls to `place_order()` for all accounts

---

### **API/Dashboard Files:**

6. **`api/backend.py`**
   - **Action:** MODIFY (remove demo data source logic)
   - **Lines affected:** ~30-45, ~120-140, ~220-240
   - **Changes needed:**
     - Remove `is_demo` flag in constructor
     - Remove `demo_state_path` and `demo_trades_path` attributes
     - Remove conditional logic in `_load_trades_dataframe()`
     - Remove conditional logic in `_load_state()`
     - Restore uniform PostgreSQL reads for all accounts

---

### **Launcher Scripts:**

7. **`main1.py`** (or primary launcher)
   - **Action:** MODIFY (remove demo initialization)
   - **Lines affected:** Varies depending on implementation
   - **Changes needed:**
     - Remove demo mode argument parsing
     - Remove demo operative instantiation
     - Remove injection of demo_operative into orchestrator

8. **Shell scripts** (if any: `start_bot.sh`, `launch_account.sh`, etc.)
   - **Action:** MODIFY
   - **Changes needed:**
     - Remove `--demo` flags or demo-specific launch logic

---

## 🔄 COMPONENT INTEGRATION REMOVAL

### **1. Orchestrator Integration**

**Location:** `core/orchestrator.py`

**Areas to clean:**

- **Constructor (`__init__`)**
  - Remove demo_operative parameter
  - Remove demo mode detection
  - Remove demo-specific path configuration

- **Initialization (`initialize`)**
  - Remove demo operative instantiation
  - Remove injection of shared state references

- **Signal Processing (`_search_signals`)**
  - Remove conditional demo order placement
  - Restore uniform `place_order()` calls

- **TP/SL Monitoring (`_periodic_tpsl_check`)**
  - Remove conditional monitoring logic
  - Use only `check_all_tp_sl()` for all accounts
  - Remove demo_operative.monitor_exits() calls

---

### **2. State Persistence Integration**

**Location:** `state/state_manager.py`

**Areas to clean:**

- **`save_state_local()` function**
  - Remove demo mode check at function entry
  - Remove early return for demo accounts
  - Restore uniform JSON + PostgreSQL writes for all accounts

---

### **3. Dashboard Data Source Integration**

**Location:** `api/backend.py`

**Areas to clean:**

- **Constructor**
  - Remove demo mode detection
  - Remove demo file paths

- **Trade Loading (`_load_trades_dataframe`)**
  - Remove Excel file reading for demo
  - Use PostgreSQL only for all accounts

- **State Loading (`_load_state`)**
  - Remove JSON file reading for demo
  - Use PostgreSQL only for all accounts

---

## 🗑️ FILES TO DELETE

### **Demo-Specific Files:**

1. **`core/demo_operative.py`** ← Primary demo module
2. **`persistence/bot_files_01/demo_state_01.json`** ← Demo state file (if exists)
3. Any demo-specific test files or scripts

**Note:** Keep `bot_trades_01.xlsx` if you want to preserve demo trade history for analysis.

---

## 🔍 SEARCH & VERIFY

After making changes, search the entire codebase for residual demo references:

### **Search Commands:**

```bash
# Search for demo-related code
grep -r "demo_operative" --include="*.py" .
grep -r "DEMO_MODE" --include="*.py" .
grep -r "is_demo" --include="*.py" .
grep -r "\[DEMO\]" --include="*.py" .
grep -r "simulated.*True" --include="*.py" .

# Search for demo file paths
grep -r "demo_state" --include="*.py" .
grep -r "demo_trades" --include="*.py" .

# Search for demo-specific logic
grep -r "if.*demo" --include="*.py" .
grep -r "hasattr.*demo_operative" --include="*.py" .
```

### **Expected Result:**
All searches should return **zero results** after complete removal.

---

## ⚙️ CONFIGURATION CLEANUP

### **Settings File (`config/settings.py`)**

**Remove:**
- `DEMO_MODE_ACCOUNTS = ['01']` or similar
- Any demo-specific constants
- Demo-related comments

**Verify:**
- Account 01 configuration still exists in `ACCOUNTS` dict
- All accounts treated uniformly

---

### **Account Configuration**

**Keep:**
- Account 01 settings in `ACCOUNTS` dict
- Dashboard port (5099)
- Initial capital

**Change:**
- Account 01 becomes a regular account (no special treatment)
- Can be used for live trading with small capital

---

## 🧪 TESTING AFTER REMOVAL

### **1. Verify Compilation**
```bash
python3 -m py_compile core/orchestrator.py
python3 -m py_compile strategies/strategy_processor.py
python3 -m py_compile state/state_manager.py
python3 -m py_compile api/backend.py
```

### **2. Verify Imports**
```bash
python3 -c "from core.orchestrator import BotOrchestrator; print('OK')"
python3 -c "from strategies.strategy_processor import StrategyProcessor; print('OK')"
```

### **3. Dry Run Test**
```bash
# Test account 01 launches without errors
python3 main1.py --account 01 --dry-run  # if dry-run mode exists
```

### **4. Verify Database Writes**
- Launch bot on account 01
- Open a test position
- Verify trade appears in PostgreSQL `trades` table
- Verify state saved in PostgreSQL + `bot_state_01.json`

### **5. Verify Dashboard**
- Navigate to `http://localhost:5099`
- Verify data loads from PostgreSQL
- Verify positions display correctly
- Verify recent trades display correctly

---

## 🚨 CRITICAL CONSIDERATIONS

### **Data Migration**

**Before removal, decide:**

1. **Demo trade history in Excel**
   - Archive for analysis?
   - Import into PostgreSQL with `SIMULATED='YES'` flag?
   - Delete permanently?

2. **Demo state in JSON**
   - Likely can be deleted (temporary data)
   - No open positions should exist

### **Account 01 Future Use**

**After removal, account 01 will:**
- Execute REAL broker orders
- Write to PostgreSQL
- Operate identically to accounts E1 and 00

**Options:**
1. Keep as low-capital test account ($117 initial capital)
2. Increase capital and use as production account
3. Deactivate by removing API credentials

### **Rollback Plan**

**If you need to restore demo mode:**
1. Keep backup of all modified files
2. Git commit before making changes: `git commit -am "Pre demo-removal checkpoint"`
3. Demo mode can be re-implemented from this conversation transcript

---

## 📊 REMOVAL CHECKLIST

### **Phase 1: Code Modifications**
- [ ] Remove demo logic from `orchestrator.py`
- [ ] Remove demo bypass from `state_manager.py`
- [ ] Remove demo routing from `strategy_processor.py`
- [ ] Remove demo data sources from `backend.py`
- [ ] Remove demo config from `settings.py`
- [ ] Update launcher scripts

### **Phase 2: File Cleanup**
- [ ] Delete `core/demo_operative.py`
- [ ] Delete demo state JSON files
- [ ] Archive or delete demo Excel trades

### **Phase 3: Verification**
- [ ] Run all grep searches (zero results expected)
- [ ] Verify Python compilation
- [ ] Test bot launch on account 01
- [ ] Verify PostgreSQL writes
- [ ] Verify dashboard displays correctly

### **Phase 4: Documentation**
- [ ] Update README if demo mode was documented
- [ ] Update architecture diagrams
- [ ] Update deployment guides

---

## 🔄 ALTERNATIVE: DISABLE INSTEAD OF REMOVE

**If you want to keep the code but disable demo mode:**

### **Minimal Changes:**

1. **`config/settings.py`**
   ```python
   DEMO_MODE_ACCOUNTS = []  # Empty list = no demo accounts
   ```

2. **Launcher**
   - Remove `--demo` flag usage
   - Never pass demo_operative to orchestrator

**Result:**
- Code remains intact (can re-enable later)
- Account 01 operates as normal account
- No demo functionality active

---

## 📝 MIGRATION TIMELINE ESTIMATE

| Phase | Tasks | Estimated Time |
|-------|-------|----------------|
| **Planning** | Review code, identify dependencies | 30 min |
| **Code Removal** | Modify 6 files, delete 1 file | 1-2 hours |
| **Testing** | Compilation, imports, dry runs | 30 min |
| **Integration Testing** | Launch bot, verify DB writes, test dashboard | 1 hour |
| **Validation** | Monitor live for 1-2 candle cycles | 2-4 hours |
| **Total** | | **5-8 hours** |

---

## 🎓 LESSONS LEARNED FROM DEMO MODE

**What worked well:**
- ✅ Clean separation from production code
- ✅ Minimal performance overhead
- ✅ Easy to validate LAB backtests
- ✅ No risk to live accounts

**What could be improved:**
- Dual persistence layer (JSON + Excel) was complex
- Dashboard integration required conditional logic
- State management needed special handling

**Recommendation for future:**
- If re-implementing, use a unified persistence abstraction layer
- Single data source with `simulated` flag in database
- Cleaner separation of concerns

---

## 📞 SUPPORT

**If you encounter issues during removal:**
1. Check this conversation transcript for context
2. Review git history for changes
3. Test incrementally (one file at a time)
4. Verify PostgreSQL schema supports all required fields

---

## ✅ COMPLETION CRITERIA

**Demo mode is fully removed when:**

1. ✅ No references to `demo_operative` in codebase
2. ✅ No demo-specific file paths in code
3. ✅ All accounts use identical persistence logic
4. ✅ Dashboard reads from PostgreSQL only
5. ✅ Account 01 executes real broker orders
6. ✅ No demo-related logs in output
7. ✅ All tests pass
8. ✅ Bot runs for 24+ hours without issues

---

**END OF GUIDE**

*This document provides a high-level roadmap. Actual implementation may require adjustments based on your specific codebase structure and requirements.*
