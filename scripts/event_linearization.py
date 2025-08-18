#!/usr/bin/env python3
"""
Simple event linearization for LUMIA XML files.

Creates string representations for events that are not of type "note".
Format: name | value | units (where value and units are optional)
"""

import xml.etree.ElementTree as ET
import argparse
import re


def linearize_event(event_element, delimiter=" | ", filter_brackets=False):
    """
    Convert a LUMIA XML event element to a linearized string.
    
    Args:
        event_element: XML element representing an event
        delimiter: String to use as delimiter between parts
        filter_brackets: If True, remove bracketed text like [#/volume] from name and value
        
    Returns:
        str: Linearized representation in format "name{delimiter}value{delimiter}units"
             where value and units are optional
    """
    # Skip note events
    event_type = event_element.get("type", "")
    if event_type == "note":
        return None
    
    # Extract core components
    name = event_element.get("name", "Unknown")
    value = (event_element.text or "").strip()
    unit = event_element.get("unit", "")
    
    # Filter brackets if requested
    if filter_brackets:
        name = re.sub(r'\[[^\]]*\]', '', name).strip()
        value = re.sub(r'\[[^\]]*\]', '', value).strip()
        unit = re.sub(r'\[[^\]]*\]', '', unit).strip()
        
        # Replace multiple whitespaces with single space
        name = re.sub(r'\s+', ' ', name).strip()
        value = re.sub(r'\s+', ' ', value).strip()
        unit = re.sub(r'\s+', ' ', unit).strip()
    
    # Build linearization
    parts = [name]
    
    # Add value if it exists and is not empty
    if value:
        parts.append(value)
    
    # Add unit if it exists and is not empty
    if unit:
        parts.append(unit)
    
    return delimiter.join(parts)


def process_lumia_xml(xml_content, delimiter=" | ", filter_brackets=False):
    """
    Process LUMIA XML content and return linearized events.
    
    Args:
        xml_content (str): Raw XML content
        delimiter (str): Delimiter to use between event parts
        filter_brackets (bool): If True, remove bracketed text from event parts
        
    Returns:
        list: List of (timestamp, linearized_event) tuples
    """
    root = ET.fromstring(xml_content)
    linearized_events = []
    
    # Find all events across all encounters
    for encounter in root.findall("encounter"):
        events_section = encounter.find("events")
        if events_section is not None:
            for entry in events_section.findall("entry"):
                timestamp = entry.get("timestamp", "")
                
                for event in entry.findall("event"):
                    linearized = linearize_event(event, delimiter, filter_brackets)
                    if linearized is not None:  # Skip note events
                        linearized_events.append((timestamp, linearized))
    
    return linearized_events


def main():
    """Demo that loads an XML file and prints linearized events."""
    parser = argparse.ArgumentParser(description="Linearize LUMIA XML events")
    parser.add_argument("xml_file", help="Path to LUMIA XML file")
    parser.add_argument("--limit", type=int, default=50, 
                       help="Maximum number of timestamps to display (default: 50)")
    parser.add_argument("--delimiter", "-d", default=" | ",
                       help="Delimiter to use between event parts (default: ' | ')")
    parser.add_argument("--list-symbol", "-l", default="\t",
                       help="Symbol to use for list items (default: '\\t')")
    parser.add_argument("--filter-brackets", "-f", action="store_true",
                       help="Remove bracketed text like [#/volume] from event names and values")
    
    args = parser.parse_args()
    
    try:
        # Load XML file
        with open(args.xml_file, 'r', encoding='utf-8') as f:
            xml_content = f.read()
        
        # Process and linearize events
        events = process_lumia_xml(xml_content, args.delimiter, args.filter_brackets)
        
        # Group events by timestamp
        from collections import defaultdict
        events_by_timestamp = defaultdict(list)
        for timestamp, event in events:
            events_by_timestamp[timestamp].append(event)
        
        # Sort timestamps
        sorted_timestamps = sorted(events_by_timestamp.keys())
        
        total_events = len(events)
        total_timestamps = len(sorted_timestamps)
        
        print(f"Found {total_events} non-note events across {total_timestamps} timestamps in {args.xml_file}")
        print("=" * 80)
        
        # Display events grouped by timestamp (limited by --limit parameter)
        for i, timestamp in enumerate(sorted_timestamps[:args.limit]):
            print(f"{timestamp}")
            for event in events_by_timestamp[timestamp]:
                print(f"{args.list_symbol}{event}")
            print()  # Empty line between timestamp groups
        
        if total_timestamps > args.limit:
            remaining_timestamps = total_timestamps - args.limit
            remaining_events = sum(len(events_by_timestamp[ts]) for ts in sorted_timestamps[args.limit:])
            print(f"... and {remaining_events} more events across {remaining_timestamps} more timestamps")
            
    except FileNotFoundError:
        print(f"Error: File '{args.xml_file}' not found")
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
