# Data Loading

!!! abstract "TL;DR"

    - A [`Dataset`][abstract_dataloader.spec.Dataset] consists of one or more traces.
    - A [`Trace`][abstract_dataloader.spec.Trace] consists of one or more simultaneously recorded sensors, which are asynchronously recorded and synchronized with a [`Synchronization`][abstract_dataloader.spec.Synchronization] protocol.
    - A [`Sensor`][abstract_dataloader.spec.Sensor] is a synchronous time series of sensor data, with associated [`Metadata`][abstract_dataloader.spec.Metadata].

## Dataset Hierarchy

Multimodal datasets generally operate using the following pattern:

- You construct a data collection system consisting of multiple sensors, which can record data simultaneously.
- The system has a way of starting a recording, which causes all of the sensors to record data, with some kind of metadata indicating which session the data came from. This session is sometimes referred to as a "trace" (also "sequence", "session", etc).
- You offload the data from the collection system, and organize them into folders by trace. You then collect many such traces, and organize them into a "dataset."

To implement a dataloader specification based on this common pattern, we start from a dataset, trace, and sensor-based architecture:

``` 
┌─────────────────────────────────────────────────────────────┐
│Dataset                                                      │
│┌─────────────────────────┐┌─────────────────────────┐       │
││Trace 1                  ││Trace 2                  │       │
││┌────────┐┌────────┐     ││┌────────┐┌────────┐     │       │
│││Sensor 1││Sensor 2│ ... │││Sensor 1││Sensor 2│ ... │  ...  │
││└────────┘└────────┘     ││└────────┘└────────┘     │       │
│└─────────────────────────┘└─────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

- A [`Sensor`][abstract_dataloader.spec.Sensor] consists of all of the data recorded by a given modality in a single recording, with a modality being defined by one or possibly multiple feature streams which make sense to be loaded together. The [`Sensor`][abstract_dataloader.spec.Sensor] keeps track of its own metadata, and has some way of loading the underlying data; the data also expected to be a single synchronous time series.
- Multiple [`Sensor`][abstract_dataloader.spec.Sensor] streams are recorded simultaneously to make up a [`Trace`][abstract_dataloader.spec.Trace], which in turn has some way of loading data across its constituent sensors.
- While a [`Trace`][abstract_dataloader.spec.Trace] can already be useful on its own (e.g. for "scene" processing pipelines like [3D reconstruction](https://docs.nerf.studio/)), many traces are then collected into a [`Dataset`][abstract_dataloader.spec.Dataset], which in turn is charged with loading data from the correct trace on each load call.

## Time Synchronization

Next, we address time synchronization between sensor streams which are "asynchronous" - that is, potentially have different data collection periods, or even have no regular period at all.

At a high level, all approaches for dealing with asynchronous time series data can be abstracted as an algorithm which takes asynchronous time stamps in, and returns instructions for how to load synchronized samples. We define this procedure using a [`Synchronization`][abstract_dataloader.spec.Synchronization] spec, which maps global trace indices to sensor indices:

``` 
┌─────────────────────────────────────────────────────────┐
│Trace: Map[int -> multimodal sample]                     │
│┌───────────────────────┐┌───────────────────────┐       │
││Sensor 1               ││Sensor 2               │       │
││┌─────────────────────┐││┌─────────────────────┐│       │
│││Metadata             ││││Metadata             ││       │
│││┌──────────────┐     ││││┌──────────────┐     ││       │
││││Timestamps:f64│ ... │││││Timestamps:f64│ ... ││  ...  │
│││└──────┬───────┘     ││││└──────┬───────┘     ││       │
││└───────┼─────────────┘││└───────┼─────────────┘│       │
│└────────┼──────────────┘└────────┼──────────────┘       │
│         ▼                        ▼                      │
│┌───────────────────────────────────────────────────────┐│
││                    Synchronization                    ││
│└────────┬────────────────────────┬─────────────────────┘│
│         ▼                        ▼                      │
│┌───────────────────────┐┌───────────────────────┐       │
││Index: trace -> sensor ││Index: trace -> sensor │       │
│└───────────────────────┘└───────────────────────┘       │
└─────────────────────────────────────────────────────────┘
```

!!! info "Trace Indexing"

    We standardize the trace index as an array of integers, i.e. `index[i]` indicates the map from trace index
    `i` to sensor index. This implies that more advanced indexing methods, e.g. returning multiple observations, need to be implemented externally at the trace and/or sensor level. We may revisit this assumption in the future.

!!! question "Timestamps: Why `float64`?"

    We standardize timestamps as float64 epoch times, which yields a precision of [better than 1us](https://www.leebutterman.com/2021/02/01/store-your-unix-epoch-times-as-float64.html). While there are probably some use cases which demand better precision, other solutions such as using a "relative" epoch time can also be entertained to increase this precision. In any case, I do not believe that the `uint64` epoch-in-nanoseconds convention provides any substantial benefits here.

## Indexing

One of the downsides of our `dataset => trace => sensor` hierarchy is the need to implement global (dataset) to local (trace) index translation, which is not needed in monolithic dataloaders.

``` 
              index
                │
┌───────────────┼────────────────┐
│Dataset        ▼                │
│┌──────────────────────────────┐│
││          Index Map           ││
│└───┬────────┬─────────────┬───┘│
│    ▼?       ▼?            ▼?   │
│┌───────┐┌───────┐     ┌───────┐│
││Trace 1││Trace 2│ ... │Trace N││
│└───┬───┘└───┬───┘     └───┬───┘│
└────┼────────┼─────────────┼────┘
     └────────┴─┬───────────┘
                ▼
              sample
```

Fortunately, aside from a negligible performance overhead, in addition to presenting a cleaner abstraction, this approach provides substantial memory advantages:

- Index lookups are cheap. Using [np.searchsorted](https://numpy.org/doc/2.2/reference/generated/numpy.searchsorted.html) (see [`abstract.Dataset`][abstract_dataloader.abstract.Dataset]), the global to local lookup latency is on the order of microseconds even with 100k traces.
- This approach saves memory. Assuming that metadata for the entire dataset will be loaded into memory, and assuming that the instructions for fetching each sample within each trace can be calculated without storing additional metadata (e.g. `trace/data/blob_{index}.npz`), no additional memory usage is required, while a global index-based system would likely require at least 4 bytes `int32` for each sample. In a more likely case, 10s of bytes would be required to store a file path, which can add up.
