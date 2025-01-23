import com.crawljax.browser.EmbeddedBrowser;
import com.crawljax.core.CrawljaxRunner;
import com.crawljax.core.configuration.BrowserConfiguration;
import com.crawljax.core.configuration.CrawlRules;
import com.crawljax.core.configuration.CrawljaxConfiguration;
import com.crawljax.core.configuration.InputSpecification;
import com.crawljax.core.state.Identification;
import com.crawljax.forms.FormInput;
import com.crawljax.plugins.crawloverview.CrawlOverview;
import state_abstraction_function.Word2VecEmbeddingStateVertexFactory;

import io.github.bonigarcia.wdm.WebDriverManager;

import java.util.concurrent.TimeUnit;

public class RunCrawler {
    private static final long WAIT_TIME_AFTER_EVENT = 500; //500
    private static final long WAIT_TIME_AFTER_RELOAD = 500; //500

//    private static final String URL = "https://www.york.ac.uk/teaching/cws/wws/webpage1.html";
    private static final String URL = "http://localhost:3000/claroline11110/";
    private static final String APP_NAME = "claroline";
    private static final int maxCrawlTime = 60;

    public static void main(String[] args) throws Exception {
        System.out.println("================== Starting Crawler for " + APP_NAME + " and maximum crawl time " + maxCrawlTime + "min ==================");
        WebDriverManager.chromedriver().clearDriverCache().clearResolutionCache();

        CrawljaxConfiguration.CrawljaxConfigurationBuilder builder = CrawljaxConfiguration.builderFor(URL);
//      1. set crawl rules
        builder.crawlRules().setFormFillMode(CrawlRules.FormFillMode.RANDOM);
        builder.crawlRules().clickDefaultElements();
        builder.crawlRules().crawlHiddenAnchors(true);
        builder.crawlRules().crawlFrames(false);
        builder.crawlRules().clickElementsInRandomOrder(false);
        builder.setUnlimitedCrawlDepth();
        builder.setUnlimitedStates();
        builder.setMaximumRunTime(maxCrawlTime, TimeUnit.MINUTES);

//         5. setup abstract function to be used
        builder.setStateVertexFactory(new Word2VecEmbeddingStateVertexFactory()); //comment in for using Transformers SAF

        // set timeouts
        builder.crawlRules().waitAfterReloadUrl(WAIT_TIME_AFTER_RELOAD, TimeUnit.MILLISECONDS);
        builder.crawlRules().waitAfterEvent(WAIT_TIME_AFTER_EVENT, TimeUnit.MILLISECONDS);

        builder.setBrowserConfig(new BrowserConfiguration(EmbeddedBrowser.BrowserType.CHROME_HEADLESS, 1));

        // CrawlOverview
        builder.addPlugin(new CrawlOverview());

        // add input respective for the app used (e.g. login information)
        builder.crawlRules().setInputSpec(Helper.setInputField(APP_NAME));

        CrawljaxRunner crawljax = new CrawljaxRunner(builder.build());
        crawljax.call();
    }

    private static class Helper {
        public static InputSpecification setInputField(String appName){
            InputSpecification input = new InputSpecification();
            switch (appName){
                case "addressbook":
                    // necessary for logging in
                    input.inputField(new FormInput(FormInput.InputType.INPUT, new Identification(Identification.How.name, "user"))).inputValues("admin");
                    input.inputField(new FormInput(FormInput.InputType.PASSWORD, new Identification(Identification.How.name, "pass"))).inputValues("secret");
                    System.out.println("Set input fields for addressbook");
                    break;
                case "claroline":
                    input.inputField(new FormInput(FormInput.InputType.INPUT, new Identification(Identification.How.name, "login"))).inputValues("admin");
                    input.inputField(new FormInput(FormInput.InputType.PASSWORD, new Identification(Identification.How.name, "password"))).inputValues("admin");
                    System.out.println("Set input fields for addressbook");
                    break;
                default:
                    System.out.println(APP_NAME + " not implemented yet");
                    break;
            }
            return input;
        }
    }
}
