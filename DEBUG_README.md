# 🔍 LEOPARTS Debugging System

This comprehensive debugging system captures **EVERYTHING** in raw detail during the extraction process. You can see every single step, input, output, and intermediate state.

## 🚀 Quick Start

### 1. Access Debug Interfaces

**Main Debug Interface:**
- Click "🔍 Debug Interface" in the sidebar
- View real-time logs with filtering and search
- See step-by-step execution details

**Debug Summary Dashboard:**
- Click "📊 Debug Summary" in the sidebar  
- Get overview statistics and performance metrics
- Analyze errors and recent activity

**Mini Debug Widget:**
- Available on the extraction page
- Quick access to current state and logs
- No need to leave the extraction page

### 2. What Gets Logged

The system captures **EVERYTHING** including:

#### 🔄 **Process Steps**
- Page loads and initialization
- User actions (button clicks, form inputs)
- Session state changes
- Function calls and returns

#### 🌐 **Web Scraping**
- Part number processing
- Cache hits/misses
- HTML content retrieval
- Scraping success/failure
- Timing and performance

#### 🤖 **LLM Interactions**
- Request details (prompts, models, parameters)
- Response content and timing
- Token usage and latency
- Rate limiting and errors

#### 📄 **PDF Processing**
- Context retrieval from vector store
- Chunk selection and content
- PDF chain inputs and outputs
- Processing time and success rates

#### 🔍 **Extraction Steps**
- Stage 1 (Web) extraction attempts
- Stage 2 (PDF) fallback processing
- JSON parsing and validation
- Success/failure for each attribute
- Raw outputs and error messages

#### 📊 **Data Transformations**
- JSON parsing attempts
- DataFrame creation and manipulation
- Session state updates
- Results processing

#### ⚡ **Performance Metrics**
- Function execution times
- LLM response latencies
- Overall process timing
- Resource usage

## 📋 Debug Log Format

Each log entry includes:

```
============================================================
STEP 1 - INFO - 2024-01-15T10:30:45.123456
============================================================
MESSAGE: Starting web scraping for part ABC123
CONTEXT: {"step": "web_scraping_start", "part_number": "ABC123"}
DATA: {
  "part_number": "ABC123",
  "cache_status": "miss",
  "timestamp": "2024-01-15T10:30:45.123456"
}
============================================================
```

## 🛠️ Using the Debug System

### Real-Time Monitoring

1. **Start an extraction process**
2. **Open Debug Summary** in another tab
3. **Watch real-time updates** as the process runs
4. **Analyze performance** and identify bottlenecks

### Error Investigation

1. **Check Debug Summary** for error counts
2. **Expand error entries** to see full context
3. **Review raw outputs** and error messages
4. **Trace execution flow** step by step

### Performance Analysis

1. **Monitor timing metrics** in Debug Summary
2. **Compare Stage 1 vs Stage 2** performance
3. **Identify slow operations** and bottlenecks
4. **Track LLM response times** and rate limits

### Data Flow Tracking

1. **Follow data transformations** step by step
2. **See input/output** for each LLM call
3. **Monitor JSON parsing** success/failure
4. **Track session state** changes

## 🔧 Debug Methods Available

### Basic Logging
```python
debug_logger.info("Message", data={"key": "value"}, context={"step": "description"})
debug_logger.warning("Warning message", data=data, context=context)
debug_logger.error("Error message", data=data, context=context)
debug_logger.debug("Debug message", data=data, context=context)
```

### Specialized Logging
```python
# LLM interactions
debug_logger.llm_request("prompt", "model", temperature, max_tokens, context)
debug_logger.llm_response("model", "response", tokens_used, latency, context)

# Web scraping
debug_logger.web_scraping("url", "html_content", "table_data", context)

# PDF processing
debug_logger.pdf_processing("filename", page_count, "text_content", context)

# Extraction steps
debug_logger.extraction_step("attribute", "source", input_data, output_data, success, context)

# Performance
debug_logger.performance("operation", duration, context)

# Exceptions
debug_logger.exception(exception, context)
```

### Session State Logging
```python
debug_logger.session_state("variable_name", value, context)
log_streamlit_state()  # Log all session state at once
```

### Data Transformations
```python
debug_logger.data_transformation("operation", input_data, output_data, context)
log_json_parsing("original_string", parsed_result, success)
```

## 📊 Debug Interface Features

### Filtering and Search
- **Log Level Filter:** Show only INFO, WARNING, ERROR, or DEBUG
- **Message Search:** Find entries containing specific text
- **Step Filtering:** Focus on specific process steps

### Real-Time Updates
- **Auto-refresh:** Automatically update every 5 seconds
- **Manual refresh:** Click refresh button for immediate update
- **Live monitoring:** Watch extraction process in real-time

### Data Export
- **Raw log view:** See complete log file content
- **JSON export:** Download structured log data
- **Session state dump:** Export current application state

## 🎯 Common Debug Scenarios

### 1. **Extraction Failing**
- Check Debug Summary for error counts
- Review error entries for specific failure reasons
- Look at LLM response content and parsing
- Verify input data quality

### 2. **Slow Performance**
- Monitor timing metrics in Debug Summary
- Identify slow LLM calls or web scraping
- Check for rate limiting or timeouts
- Analyze context retrieval performance

### 3. **Unexpected Results**
- Trace data flow through each step
- Check JSON parsing success/failure
- Review raw LLM outputs
- Verify attribute mapping and extraction

### 4. **Session State Issues**
- Monitor session state changes
- Check for missing or corrupted data
- Verify state persistence across page reloads
- Debug initialization problems

## 🔍 Advanced Debugging

### Function Decorators
```python
@debug_log_function
def my_function(arg1, arg2):
    # Function calls and returns are automatically logged
    return result
```

### Context Managers
```python
with DebugTimer("operation_name", context={"step": "description"}):
    # Operation timing is automatically logged
    perform_operation()
```

### Custom Debug Points
```python
# Add custom debug points anywhere in your code
debug_logger.info("Custom debug point", data={"custom": "data"}, context={"location": "function_name"})
```

## 📈 Performance Monitoring

The debug system automatically tracks:

- **Function execution times**
- **LLM response latencies**
- **Web scraping duration**
- **PDF processing time**
- **Overall extraction duration**
- **Rate limiting frequency**
- **Error rates and types**

## 🚨 Troubleshooting

### Debug Log Not Appearing
1. Check if `debug_log.txt` file is being created
2. Verify debug logger is imported correctly
3. Ensure debug calls are being executed
4. Check file permissions and disk space

### Performance Issues
1. Monitor timing metrics in Debug Summary
2. Look for slow operations in recent activity
3. Check for rate limiting or timeouts
4. Analyze LLM response times

### Missing Information
1. Add more debug calls to capture missing data
2. Use `log_streamlit_state()` to capture session state
3. Add context information to existing debug calls
4. Check if specific steps are being executed

## 📝 Best Practices

1. **Use descriptive context:** Always include meaningful context in debug calls
2. **Log at appropriate levels:** Use INFO for normal flow, WARNING for issues, ERROR for failures
3. **Include relevant data:** Log input/output data that helps with debugging
4. **Monitor in real-time:** Use Debug Summary during active development
5. **Clean up old logs:** Clear debug logs periodically to avoid file size issues

## 🎉 Summary

This debugging system gives you **complete visibility** into every aspect of the extraction process. You can:

- ✅ See every step in real-time
- ✅ Track all data transformations
- ✅ Monitor performance metrics
- ✅ Debug errors and issues
- ✅ Analyze LLM interactions
- ✅ Verify data flow and state

**When you say "EVERYTHING" - this system captures EVERYTHING!** 🚀 