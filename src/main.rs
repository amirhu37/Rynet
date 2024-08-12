enum DataType {
    Integer(i32),
    Float(f64),
    Text(String),
    Boolean(bool),
    // You can add more variants as needed
}

fn main() {
    // Example usage of the enum
    let my_integer = DataType::Integer(42);
    let my_float = DataType::Float(3.14);
    let my_text = DataType::Text(String::from("Hello, Rust!"));
    let my_boolean = DataType::Boolean(true);

    // Handling the enum variants
    match my_integer {
        DataType::Integer(val) => println!("Integer: {}", val),
        DataType::Float(val) => println!("Float: {}", val),
        DataType::Text(val) => println!("Text: {}", val),
        DataType::Boolean(val) => println!("Boolean: {}", val),
    }
}
