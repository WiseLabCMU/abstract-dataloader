# Using an Abstract Dataloader

A fully implemented Abstract Dataloader (ADL) - compliant system should include the following parts:

## [`Sensor`][abstract_dataloader.spec.Sensor]


##  [`Trace`][abstract_dataloader.spec.Trace]

!!! info

    A [`Trace`][abstract_dataloader.spec.Trace] implementation should take (and [`abstract.Trace`][abstract_dataloader.abstract.Trace] does take) a [`Synchronization`][abstract_dataloader.spec.Synchronization] policy as an argument. A few generic implementations are included with [`abstract_dataloader.generic`][abstract_dataloader.generic]:

    | Class | Description |
    | ----- | ----------- |
    | [`Empty`][abstract_dataloader.generic.Empty] | a no-op for intializing a trace without any synchronization (i.e., just as a container of sensors). |
    | [`Nearest`][abstract_dataloader.generic.Nearest] | find the nearest measurement for each sensor relative to the reference sensor's measurements. |
    | [`Next`][abstract_dataloader.generic.Next] | find the next measurement for each sensor relative to the reference sensor's measurements. |

## [`Dataset`][abstract_dataloader.spec.Dataset]

## (Optional) 
