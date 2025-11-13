# Tips and tricks

### `optimistix` and `lineax` solver options

By default `lineax` and `optimistix` check that the solve was successful. This extra check that the return doesn't have NaNs etc., may induce extra costs. This can be disabled by passing `solver(..., throw=False)`.